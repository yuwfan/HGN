from logging import basicConfig, getLogger, INFO
from sys import argv
from os.path import join as os_path_join, exists as os_path_exists
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torch import (
    load as torch_load,
    save as torch_save,
)
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import get_rank
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from apex.amp import initialize, scale_loss, master_params
from csr_mhqa.argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
from csr_mhqa.data_processing import DataHelper
from csr_mhqa.utils import load_encoder_model, get_optimizer, compute_loss, eval_model, MODEL_CLASSES
from models.HGN import HierarchicalGraphNetwork

basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=INFO)
logger = getLogger(__name__)

#########################################################################
# Initialize arguments
##########################################################################
parser = default_train_parser()

logger.info("IN CMD MODE")
args_config_provided = parser.parse_args(argv[1:])
if args_config_provided.config_file is not None:
    argv = json_to_argv(args_config_provided.config_file) + argv[1:]
else:
    argv = argv[1:]
args = parser.parse_args(argv)
args = complete_default_train_parser(args)

divider: str = '-' * 100
logger.info(f'{divider}\nInput Argument Information\n{divider}')
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
exp_name: str = args.exp_name
cached_config_file: str = os_path_join(exp_name, 'cached_config.bin')
if os_path_exists(cached_config_file):
    cached_config = torch_load(cached_config_file)
    encoder_path: str = os_path_join(exp_name, cached_config['encoder'])
    model_path: str = os_path_join(exp_name, cached_config['model'])
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
model_type: str = args.model_type
encoder_name_or_path: str = args.encoder_name_or_path
encoder, _ = load_encoder_model(encoder_name_or_path, model_type)
model = HierarchicalGraphNetwork(config=args)

if encoder_path is not None:
    encoder.load_state_dict(torch_load(encoder_path))
if model_path is not None:
    model.load_state_dict(torch_load(model_path))

device = args.device
encoder.to(device)
model.to(device)

_, _, tokenizer_class = MODEL_CLASSES[model_type]
tokenizer = tokenizer_class.from_pretrained(encoder_name_or_path,
                                            do_lower_case=args.do_lower_case)

#########################################################################
# Evalaute if resumed from other checkpoint
##########################################################################
dev_gold_file: str = args.dev_gold_file
if encoder_path is not None and model_path is not None:
    output_pred_file: str = os_path_join(exp_name, 'prev_checkpoint.pred.json')
    output_eval_file: str = os_path_join(exp_name, 'prev_checkpoint.eval.txt')
    prev_metrics, prev_threshold = eval_model(args, encoder, model,
                                              dev_dataloader, dev_example_dict, dev_feature_dict,
                                              output_pred_file, output_eval_file, dev_gold_file)
    logger.info("Best threshold for prev checkpoint: {}".format(prev_threshold))
    for key, val in prev_metrics.items():
        logger.info("{} = {}".format(key, val))

#########################################################################
# Get Optimizer
##########################################################################
gradient_accumulation_steps: int = args.gradient_accumulation_steps
max_steps: float = args.max_steps
num_train_epochs: int = args.num_train_epochs
if max_steps > 0:
    t_total: int = max_steps
    num_train_epochs: int = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
else:
    t_total: int = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

optimizer = get_optimizer(encoder, model, args, learning_rate, remove_pooler=False)
fp16: bool = args.fp16
if fp16:
    models, optimizer = initialize([encoder, model], optimizer, opt_level=args.fp16_opt_level)
    assert len(models) == 2
    encoder, model = models

# Distributed training (should be after apex fp16 initialization)
local_rank: int = args.local_rank
if local_rank != -1:
    encoder = DistributedDataParallel(encoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=args.warmup_steps,
                                            num_training_steps=t_total)

#########################################################################
# launch training
##########################################################################
global_step = 0
loss_name = ["loss_total", "loss_span", "loss_type", "loss_sup", "loss_ent", "loss_para"]
tr_loss, logging_loss = [0] * len(loss_name), [0]* len(loss_name)
main_thread: bool = local_rank in [-1, 0]
if main_thread:
    tb_writer = SummaryWriter(exp_name)

encoder.zero_grad()
model.zero_grad()

disable_tqdm: bool = not main_thread
max_grad_norm: float = args.max_grad_norm
logging_steps: int = args.logging_steps
use_segment_idxs: bool = model_type in ['bert', 'xlnet']
train_iterator: tqdm[int] = trange(start_epoch, start_epoch+int(num_train_epochs), desc="Epoch", disable=disable_tqdm)
n_gpu: int = args.n_gpu
for epoch in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=disable_tqdm)
    train_dataloader.refresh()
    dev_dataloader.refresh()

    for step, batch in enumerate(epoch_iterator):
        encoder.train()
        model.train()

        context_mask = batch['context_mask']
        inputs = {'input_ids':      batch['context_idxs'],
                  'attention_mask': context_mask,
                  'token_type_ids': batch['segment_idxs'] if args.model_type in ['bert', 'xlnet'] else None}  # XLM don't use segment_ids

        batch['context_encoding'] = encoder(**inputs)[0]
        batch['context_mask'] = context_mask.float().to(device)
        del context_mask
        start, end, q_type, paras, sents, ents, _, _ = model(batch, return_yp=True)

        loss_list = compute_loss(args, batch, start, end, paras, sents, ents, q_type)
        del batch

        if n_gpu > 1:
            for loss in loss_list:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
        if gradient_accumulation_steps > 1:
            for loss in loss_list:
                loss = loss / gradient_accumulation_steps
        if fp16:
            with scale_loss(loss_list[0], optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(master_params(optimizer), max_grad_norm)
        else:
            loss_list[0].backward()
            clip_grad_norm_(encoder.parameters(), max_grad_norm)
            clip_grad_norm_(model.parameters(), max_grad_norm)

        for idx in range(len(loss_name)):
            loss_item = loss_list[idx]
            if not isinstance(loss_item, int):
                tr_loss[idx] += loss_item.data.item()
            else:
                tr_loss[idx] += loss_item

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            encoder.zero_grad()
            model.zero_grad()
            global_step += 1

            if main_thread and logging_steps > 0 and global_step % logging_steps == 0:
                avg_loss = [ (_tr_loss - _logging_loss) / (logging_steps*gradient_accumulation_steps)
                             for (_tr_loss, _logging_loss) in zip(tr_loss, logging_loss)]

                loss_str = "step[{0:6}] " + " ".join(['%s[{%d:.5f}]' % (loss_name[i], i+1) for i in range(len(avg_loss))])
                logger.info(loss_str.format(global_step, *avg_loss))

                # tensorboard
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                for i in range(len(loss_name)):
                    tb_writer.add_scalar(loss_name[i], (tr_loss[i]- logging_loss[i])/(logging_steps * gradient_accumulation_steps), global_step)
                logging_loss = tr_loss.copy()
        if max_steps > 0 and global_step > max_steps:
            epoch_iterator.close()
            break
    if local_rank == -1 or get_rank() == 0:
        output_pred_file: str = os_path_join(exp_name, f'pred.epoch_{epoch+1}.json')
        output_eval_file: str = os_path_join(exp_name, f'eval.epoch_{epoch+1}.txt')
        metrics, threshold = eval_model(args, encoder, model,
                                        dev_dataloader, dev_example_dict, dev_feature_dict,
                                        output_pred_file, output_eval_file, dev_gold_file)

        if metrics['joint_f1'] >= best_joint_f1:
            best_joint_f1 = metrics['joint_f1']
            torch_save({'epoch': epoch+1,
                        'lr': scheduler.get_lr()[0],
                        'encoder': 'encoder.pkl',
                        'model': 'model.pkl',
                        'best_joint_f1': best_joint_f1,
                        'threshold': threshold},
                        os_path_join(exp_name, f'cached_config.bin')
            )
        torch_save({k: v.cpu() for k, v in encoder.state_dict().items()},
                    os_path_join(exp_name, f'encoder_{epoch+1}.pkl'))
        torch_save({k: v.cpu() for k, v in model.state_dict().items()},
                    os_path_join(exp_name, f'model_{epoch+1}.pkl'))

        for key, val in metrics.items():
            tb_writer.add_scalar(key, val, epoch)

if main_thread:
    tb_writer.close()
