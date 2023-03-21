import argparse
import numpy as np
import logging
import sys
import os
import shutil
import json

from os.path import join
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from csr_mhqa.argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
from csr_mhqa.data_processing import Example, InputFeatures, DataHelper
from csr_mhqa.utils import load_encoder_model, convert_to_tokens, hotpot_eval, MODEL_CLASSES, IGNORE_INDEX
from models.PredictionLayerOnly import *
from transformers import get_linear_schedule_with_warmup, MultiModalStructAdaptFastRoberta_v2_skipIB, AdamW
from envs import DATASET_FOLDER

import neptune.new as neptune
from neptune.new.integrations.python_logger import NeptuneHandler

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_t_p(graphqa, print_stats=False):
    params = []
    params_name = []
    params_name_frozen = []

    num_training_params = 0
    num_fronzen_params = 0
    training_params = ['text', 'predict_layer', 'graph', 'fusing']
    dict_params = {p: 0 for p in training_params}
    
    def trained_par(n, p, num_training_params, params, params_name):
        num_training_params += p.numel()
        params.append(p)
        params_name.append(n)
        return True
    
    for n, p in graphqa.named_parameters():
        trained = False
        if 'adapter' in n and not 'graph' in n and not 'text' in n or 'predict_layer' in n and 'projectionlayer' in n:
            # Add fusing layer and according projection layer to 'fusion'-tag
            dict_params['fusing'] += p.numel()
            trained = trained_par(n, p, num_training_params, params, params_name)
        elif 'adapter' in n and not 'graph' in n:
            dict_params['text'] +=  p.numel()
            trained = trained_par(n, p, num_training_params, params, params_name)
        elif 'graph' in n:
            dict_params['graph'] += p.numel()
            trained = trained_par(n, p, num_training_params, params, params_name)
        elif 'predict_layer' in n:
            dict_params['predict_layer'] += p.numel()
            trained = trained_par(n, p, num_training_params, params, params_name)
        if not trained:
            num_fronzen_params += p.numel()
            params_name_frozen.append(n)

    dict_params["frozen"] = num_fronzen_params
    return dict_params


def get_training_params(graphqa, print_stats=False):
    params = []
    params_name = []
    params_name_frozen = []

    num_training_params = 0
    num_fronzen_params = 0
    training_params = ['text', 'predict_layer', 'graph', 'fusing']
    dict_params = {p: 0 for p in training_params}
    
    def trained_par(n, p, num_training_params, params, params_name):
        num_training_params += p.numel()
        params.append(p)
        params_name.append(n)
        return True


    for n, p in graphqa.named_parameters():
        trained = False
        if 'adapter' in n and not 'graph' in n and not 'text' in n or 'predict_layer' in n and 'projectionlayer' in n:
            # Add fusing layer and according projection layer to 'fusion'-tag
            dict_params['fusing'] += p.numel()
            trained = trained_par(n, p, num_training_params, params, params_name)
        elif 'adapter' in n and not 'graph' in n:
            dict_params['text'] +=  p.numel()
            trained = trained_par(n, p, num_training_params, params, params_name)
        elif 'graph' in n:
            dict_params['graph'] += p.numel()
            trained = trained_par(n, p, num_training_params, params, params_name)
        elif 'predict_layer' in n:
            dict_params['predict_layer'] += p.numel()
            trained = trained_par(n, p, num_training_params, params, params_name)
        if not trained:
            num_fronzen_params += p.numel()
            params_name_frozen.append(n)


    if print_stats:
        num_total_params = num_training_params + num_fronzen_params
        logger.info(f"Number of training parameters: {num_training_params/1e6:.2f}M")
        run["model/weights/num_training_params"] = f"{num_training_params/1e6:.2f}M"
        logger.info(f"Number of frozen parameters: {num_fronzen_params/1e6:.2f}M")
        run["model/weights/num_fronzen_params"] = f"{num_fronzen_params/1e6:.2f}M"
        logger.info(f"Number of total parameters: {num_total_params/1e6:.2f}M")
        run["model/weights/num_total_params"] = f"{num_total_params/1e6:.2f}M"
        logger.info(f"-----------------------")
        for k, v in dict_params.items():
            logger.info(f"Number of {k} parameters: {v/1e6:.2f}M")
            run[f"model/weights/{k}_params"] = f"{v/1e6:.2f}M"       
        logger.info(f"-----------------------")
        logger.info(f"Ratio learned parameters: { num_training_params / num_fronzen_params:.2f}")
        run["model/weights/ratio_learned_params"] = f"{ num_training_params / num_fronzen_params:.2f}"
    return params_name, params


def get_optimizer(model, args, learning_rate, remove_pooler=False):
    """
    get BertAdam for encoder / classifier or BertModel
    :param model:
    :param classifier:
    :param args:
    :param remove_pooler:
    :return:
    """
    num_training_params = 0
    params_name, params = get_training_params(model, print_stats=True)
    # logger.info(f"Name of the training parameters: {params_name}")

    for p in params:
        num_training_params += p.numel()
    logger.info(f"Number of parameters in the model: {num_training_params/1e6:.2f}M")
    # logger.info(f"Name of the training parameters: {params_name}")

    no_decay = ["bias", "LayerNorm.weight"]
    weight_decay = 0
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in zip(params_name, params) if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in zip(params_name, params) if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

    return optimizer

def compute_loss(args, batch, start, end, para, sent, ent, q_type):
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    loss_span = args.ans_lambda * (criterion(start, batch['y1']) + criterion(end, batch['y2']))
    loss_type = args.type_lambda * criterion(q_type, batch['q_type'])

    sent_pred = sent.view(-1, 2)
    sent_gold = batch['is_support'].long().view(-1)
    loss_sup = args.sent_lambda * criterion(sent_pred, sent_gold.long())

    loss_ent = args.ent_lambda * criterion(ent, batch['is_gold_ent'].long())
    loss_para = args.para_lambda * criterion(para.view(-1, 2), batch['is_gold_para'].long().view(-1))

    loss = loss_span + loss_type + loss_sup + loss_ent + loss_para

    return loss, loss_span, loss_type, loss_sup, loss_ent, loss_para


def eval_model(args, model, dataloader, example_dict, feature_dict, prediction_file, eval_file, dev_gold_file, thresholds=None):
    model.eval()

    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}

    dataloader.refresh()

    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    else:
        thresholds = np.array([thresholds])
    N_thresh = len(thresholds)
    total_sp_dict = [{} for _ in range(N_thresh)]

    for batch in tqdm(dataloader, file=sys.stdout):
        with torch.no_grad():
            batch['context_mask'] = batch['context_mask'].float().to(args.device)
            start_prediction, end_prediction, type_prediction, para, sent, ent, yp1, yp2 = model(batch=batch, return_yp=True)
        type_prob = F.softmax(type_prediction, dim=1).data.cpu().numpy()
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)

        answer_type_dict.update(answer_type_dict_)
        answer_type_prob_dict.update(answer_type_prob_dict_)
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sent[:, :, 1]).data.cpu().numpy()

        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = [[] for _ in range(N_thresh)]
            cur_id = batch['ids'][i]

            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break

                for thresh_i in range(N_thresh):
                    if predict_support_np[i, j] > thresholds[thresh_i]:
                        cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[j])

            for thresh_i in range(N_thresh):
                if cur_id not in total_sp_dict[thresh_i]:
                    total_sp_dict[thresh_i][cur_id] = []

                total_sp_dict[thresh_i][cur_id].extend(cur_sp_pred[thresh_i])

    def choose_best_threshold(ans_dict, pred_file):
        best_joint_f1 = 0
        best_metrics = None
        best_threshold = 0
        for thresh_i in range(N_thresh):
            prediction = {'answer': ans_dict,
                          'sp': total_sp_dict[thresh_i],
                          'type': answer_type_dict,
                          'type_prob': answer_type_prob_dict}
            tmp_file = os.path.join(os.path.dirname(pred_file), 'tmp.json')
            with open(tmp_file, 'w') as f:
                json.dump(prediction, f)
            metrics = hotpot_eval(tmp_file, dev_gold_file)
            if metrics['joint_f1'] >= best_joint_f1:
                best_joint_f1 = metrics['joint_f1']
                best_threshold = thresholds[thresh_i]
                best_metrics = metrics
                shutil.move(tmp_file, pred_file)

        return best_metrics, best_threshold

    best_metrics, best_threshold = choose_best_threshold(answer_dict, prediction_file)
    json.dump(best_metrics, open(eval_file, 'w'))

    return best_metrics, best_threshold, answer_dict

def change_argument(args_list, args_temp):
    # argument_name = '--adapter_size'
    # new_argument = '32'
    for new_arg in args_list:
        for i, ar in enumerate(argv):
            if ar == new_arg[0]:
                argv[i+1] = new_arg[1]

    args_temp = parser.parse_args(argv)
    args_temp = complete_default_train_parser(args_temp)
    # run["model/parameters"] = vars(args)

    logger.info('-' * 100)
    logger.info('Input Argument Information')
    logger.info('-' * 100)
    args_dict = vars(args_temp)
    for a in args_dict:
        logger.info('%-28s  %s' % (a, args_dict[a]))
    return args_temp


run = neptune.init(tags=["StructAdapt", "MultiModal", "SkipIB"])
logger.addHandler(NeptuneHandler(run=run))

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
run["model/parameters"] = vars(args)

logger.info('-' * 100)
logger.info('Input Argument Information')
logger.info('-' * 100)
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

def Init_Model(ar):
    cached_config_file = join(ar.exp_name, 'cached_config.bin')
    if os.path.exists(cached_config_file):
        cached_config = torch.load(cached_config_file)
        encoder_path = join(ar.exp_name, cached_config['encoder'])
        model_path = join(ar.exp_name, cached_config['model'])
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
        learning_rate = ar.learning_rate
    return ar
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

# import ipdb; ipdb.set_trace()

# Set Encoder and Model
model = MultiModalStructAdaptFastRoberta_v2_skipIB(args)
model.to(args.device)


def calc_params(params):
    ar =change_argument(params,None)
    Init_Model(ar)
    model = MultiModalStructAdaptFastRoberta_v2_skipIB(ar)
    result = get_t_p(model, print_stats=True)
    del model
    return result


# for i in range(191,300):
#     with open("Parameter/01_29/multimodal_1.txt", 'a') as f:
#         params_dict = calc_params([('--adapter_size',str(i)), ('--hgn_hidden_size', str(i)), ('--hidden_dim', str(i))])
#         f.write(str(i)+";")
#         f.write(str(params_dict)+"\n")
#         f.close()           

def new_model(adapt = '86', hid='300'):
    a13 =change_argument([('--adapter_size',adapt), ('--hgn_hidden_size', hid), ('--hidden_dim',hid)],None)
    Init_Model(a13)
    model = MultiModalStructAdaptFastRoberta_v2_skipIB(a13)
    optimizer = get_optimizer(model, a13, learning_rate, remove_pooler=False)

# new_model(adapt = '128', hid='300')

# import ipdb; ipdb.set_trace()


# params_dict = calc_params([('--adapter_size','70'),('--hgn_hidden_size', '140')])

# #######################     Check sizes of model:   #############################
# # Step by step instructions for new Model with different size:
# # 1.) Name it and chenge args
#  a34 =change_argument([('--adapter_size','482'),('--hgn_hidden_size', '65'),('--hidden_dim', '65')],None)
# # 2.) Initialize model
# Init_Model(a34)
# # 3.) Create model
# model = MultiModalStructAdaptFastRoberta_v2(a34)
# # 4.) Output parameters
# optimizer = get_optimizer(model, a34, learning_rate, remove_pooler=False)
# params_name, params = get_training_params(model, print_stats=True)
# ################################################################################

_, _, tokenizer_class = MODEL_CLASSES[args.model_type]
tokenizer = tokenizer_class.from_pretrained(args.encoder_name_or_path,
                                            do_lower_case=args.do_lower_case)


#########################################################################
# Get Optimizer
##########################################################################
if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

optimizer = get_optimizer(model, args, learning_rate, remove_pooler=False)

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=args.warmup_steps,
                                            num_training_steps=t_total)

#########################################################################
# launch training
##########################################################################
global_step = 0
loss_name = ["loss_total", "loss_span", "loss_type", "loss_sup"]
tr_loss, logging_loss = [0] * len(loss_name), [0]* len(loss_name)
if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter(args.exp_name)

model.zero_grad()
list_few_shot_eval = np.array([100])/args.batch_size
logger.info(f"Few-shot evaluation at {list_few_shot_eval}")

best_f1 = 0
best_threshold = 0
train_iterator = trange(start_epoch, start_epoch+int(args.num_train_epochs), desc="Epoch", file=sys.stdout, disable=args.local_rank not in [-1, 0])
for epoch in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", file=sys.stdout, disable=args.local_rank not in [-1, 0])
    train_dataloader.refresh()
    dev_dataloader.refresh()

    for step, batch in enumerate(epoch_iterator):
        model.train()

        batch['context_mask'] = batch['context_mask'].float().to(args.device)
        start, end, q_type, paras, sents, ents, _, _ = model(batch=batch, return_yp=True)
        loss_list = compute_loss(args, batch, start, end, paras, sents, ents, q_type)
        del batch

        loss_list[0].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        for idx in range(len(loss_name)):
            if not isinstance(loss_list[idx], int):
                tr_loss[idx] += loss_list[idx].data.item()
            else:
                tr_loss[idx] += loss_list[idx]

            run["train/epoch/"+loss_name[idx]].log(loss_list[idx].data.item())
            run["train/epoch/lr"].log(scheduler.get_lr()[0])
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
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

        if epoch == 0 and step+1 in list_few_shot_eval:
            logger.info(f"Evaluating on dev set at step {global_step}")
            output_pred_file = os.path.join(args.exp_name, f'pred.global_step_{global_step}.json')
            output_eval_file = os.path.join(args.exp_name, f'eval.global_step_{global_step}.json')
            metrics, threshold, answer_dict = eval_model(args, model,
                                        dev_dataloader, dev_example_dict, dev_feature_dict,
                                        output_pred_file, output_eval_file, args.dev_gold_file)
            for key, value in metrics.items():
                run[f"dev/few_shot/{key}"].log(round(value*100, 2))
            run["dev/few_shot/preds"].log(answer_dict)
            model.train()

    torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                join(args.exp_name, f'model_{epoch+1}.pkl'))
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        output_pred_file = os.path.join(args.exp_name, f'pred.epoch_{epoch+1}.json')
        output_eval_file = os.path.join(args.exp_name, f'eval.epoch_{epoch+1}.json')
        metrics, threshold, answer_dict = eval_model(args, model,
                                        dev_dataloader, dev_example_dict, dev_feature_dict,
                                        output_pred_file, output_eval_file, args.dev_gold_file)
        for key, value in metrics.items():
            run[f"dev/{key}"].log(round(value*100, 2))
        run["dev/preds"].log(answer_dict)
        logger.info(f"Current f1: {metrics['f1']}, best f1: {best_f1}. Saving: {best_f1 <= metrics['f1']}")
        if metrics['f1'] >= best_f1:
            best_f1 = metrics['f1']
            best_epoch = epoch
            best_threshold = threshold
            model_path = os.path.join(args.exp_name, 'best_model.bin')
            logger.info(f"Saving model {model_path}")
            torch.save(model.state_dict(), model_path)
        model.train()


model_path = os.path.join(args.exp_name, 'best_model.bin')
model.load_state_dict(torch.load(model_path))
test_example_dict = helper.test_example_dict
test_feature_dict = helper.test_feature_dict
test_dataloader = helper.test_loader

metrics, threshold, answer_dict = eval_model(args, model,
                                test_dataloader, test_example_dict, test_feature_dict,
                                output_pred_file, output_eval_file, args.test_gold_file, best_threshold)
logger.info(f"Best threshold: {best_threshold} vs. {threshold}")
for key, value in metrics.items():
    run[f"test/{key}"] = round(value*100, 2)
run["test/preds"] = answer_dict
run.stop()