import argparse
import numpy as np
import logging
import sys

from os.path import join
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from csr_mhqa.argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
from csr_mhqa.data_processing import Example, InputFeatures, DataHelper
from csr_mhqa.utils import *

from models.HGN import *
from transformers import get_linear_schedule_with_warmup

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#########################################################################
# Initialize arguments
##########################################################################
parser = default_train_parser()

logger.info("IN CMD MODE")
args_config_provided = parser.parse_args(sys.argv[1:])
if args_config_provided.config_file is not None:
    argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
else:
    argv = sys.argv[1:]
args = parser.parse_args(argv)
args = complete_default_train_parser(args)

logger.info('-' * 100)
logger.info('Input Argument Information')
logger.info('-' * 100)
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

#########################################################################
# Read Data
##########################################################################
helper = DataHelper(gz=True, config=args)

# Set datasets
train_dataloader = helper.train_loader
dev_example_dict = helper.dev_example_dict
dev_feature_dict = helper.dev_feature_dict
dev_dataloader = helper.dev_loader

#########################################################################
# Initialize Model
##########################################################################
cached_config_file = join(args.exp_name, 'cached_config.bin')
if os.path.exists(cached_config_file):
    cached_config = torch.load(cached_config_file)
    encoder_path = join(args.exp_name, cached_config['encoder'])
    model_path = join(args.exp_name, cached_config['model'])
    learning_rate = cached_config['lr']
    start_epoch = cached_config['epoch']
    best_joint_f1 = cached_config['best_joint_f1']
    logger.info("Loading encoder from: {}".format(encoder_path))
    logger.info("Loading model from: {}".format(model_path))
else:
    encoder_path = None
    model_path = None
    start_epoch = 0
    best_joint_f1 = 0
    learning_rate = args.learning_rate

# Set Encoder and Model
encoder, _ = load_encoder_model(args.encoder_name_or_path, args.model_type)
model = HierarchicalGraphNetwork(config=args)

if encoder_path is not None:
    encoder.load_state_dict(torch.load(encoder_path))
if model_path is not None:
    model.load_state_dict(torch.load(model_path))

encoder.to(args.device)
model.to(args.device)

_, _, tokenizer_class = MODEL_CLASSES[args.model_type]
tokenizer = tokenizer_class.from_pretrained(args.encoder_name_or_path,
                                            do_lower_case=args.do_lower_case)

#########################################################################
# Evalaute if resumed from other checkpoint
##########################################################################
if encoder_path is not None and model_path is not None:
    output_pred_file = os.path.join(args.exp_name, 'prev_checkpoint.pred.json')
    output_eval_file = os.path.join(args.exp_name, 'prev_checkpoint.eval.txt')
    prev_metrics, prev_threshold = eval_model(args, encoder, model,
                                              dev_dataloader, dev_example_dict, dev_feature_dict,
                                              output_pred_file, output_eval_file, args.dev_gold_file)
    logger.info("Best threshold for prev checkpoint: {}".format(prev_threshold))
    for key, val in prev_metrics.items():
        logger.info("{} = {}".format(key, val))

#########################################################################
# Get Optimizer
##########################################################################
if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

optimizer = get_optimizer(encoder, model, args, learning_rate, remove_pooler=False)
if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    models, optimizer = amp.initialize([encoder, model], optimizer, opt_level=args.fp16_opt_level)
    assert len(models) == 2
    encoder, model = models

# Distributed training (should be after apex fp16 initialization)
if args.local_rank != -1:
    encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[args.local_rank],
                                                        output_device=args.local_rank,
                                                        find_unused_parameters=True)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank,
                                                      find_unused_parameters=True)

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=args.warmup_steps,
                                            num_training_steps=t_total)

#########################################################################
# launch training
##########################################################################
global_step = 0
loss_name = ["loss_total", "loss_span", "loss_type", "loss_sup", "loss_ent", "loss_para"]
tr_loss, logging_loss = [0] * len(loss_name), [0]* len(loss_name)
if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter(args.exp_name)

encoder.zero_grad()
model.zero_grad()

train_iterator = trange(start_epoch, start_epoch+int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
for epoch in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    train_dataloader.refresh()
    dev_dataloader.refresh()

    for step, batch in enumerate(epoch_iterator):
        encoder.train()
        model.train()

        inputs = {'input_ids':      batch['context_idxs'],
                  'attention_mask': batch['context_mask'],
                  'token_type_ids': batch['segment_idxs'] if args.model_type in ['bert', 'xlnet'] else None}  # XLM don't use segment_ids

        batch['context_encoding'] = encoder(**inputs)[0]
        batch['context_mask'] = batch['context_mask'].float().to(args.device)
        start, end, q_type, paras, sents, ents, _, _ = model(batch, return_yp=True)

        loss_list = compute_loss(args, batch, start, end, paras, sents, ents, q_type)
        del batch

        if args.n_gpu > 1:
            for loss in loss_list:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            for loss in loss_list:
                loss = loss / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss_list[0], optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss_list[0].backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        for idx in range(len(loss_name)):
            if not isinstance(loss_list[idx], int):
                tr_loss[idx] += loss_list[idx].data.item()
            else:
                tr_loss[idx] += loss_list[idx]

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            encoder.zero_grad()
            model.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                avg_loss = [ (_tr_loss - _logging_loss) / (args.logging_steps*args.gradient_accumulation_steps)
                             for (_tr_loss, _logging_loss) in zip(tr_loss, logging_loss)]

                loss_str = "step[{0:6}] " + " ".join(['%s[{%d:.5f}]' % (loss_name[i], i+1) for i in range(len(avg_loss))])
                logger.info(loss_str.format(global_step, *avg_loss))

                # tensorboard
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                for i in range(len(loss_name)):
                    tb_writer.add_scalar(loss_name[i], (tr_loss[i]- logging_loss[i])/(args.logging_steps * args.gradient_accumulation_steps), global_step)
                logging_loss = tr_loss.copy()
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        output_pred_file = os.path.join(args.exp_name, f'pred.epoch_{epoch+1}.json')
        output_eval_file = os.path.join(args.exp_name, f'eval.epoch_{epoch+1}.txt')
        metrics, threshold = eval_model(args, encoder, model,
                                        dev_dataloader, dev_example_dict, dev_feature_dict,
                                        output_pred_file, output_eval_file, args.dev_gold_file)

        if metrics['joint_f1'] >= best_joint_f1:
            best_joint_f1 = metrics['joint_f1']
            torch.save({'epoch': epoch+1,
                        'lr': scheduler.get_lr()[0],
                        'encoder': 'encoder.pkl',
                        'model': 'model.pkl',
                        'best_joint_f1': best_joint_f1,
                        'threshold': threshold},
                       join(args.exp_name, f'cached_config.bin')
            )
        torch.save({k: v.cpu() for k, v in encoder.state_dict().items()},
                    join(args.exp_name, f'encoder_{epoch+1}.pkl'))
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                    join(args.exp_name, f'model_{epoch+1}.pkl'))

        for key, val in metrics.items():
            tb_writer.add_scalar(key, val, epoch)

if args.local_rank in [-1, 0]:
    tb_writer.close()
