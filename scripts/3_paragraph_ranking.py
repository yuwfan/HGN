from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import pandas
import torch
import json

from os.path import join
from collections import Counter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

# This line must be above local package reference
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,
                          RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification)

from utils.feature_extraction import (convert_examples_to_features, output_modes, processors)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)), ())
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

logger = logging.getLogger(__name__)

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task = args.task_name
    eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d" % len(eval_dataset))
    print("  Batch size = %d" % args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    predictions = []
    ground_truth = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].detach().cpu().numpy()

        predictions.append(logits)
        ground_truth.extend([label_id.item() for label_id in label_ids])

        nb_eval_steps += 1
        if preds is None:
            preds = logits
            out_label_ids = label_ids
        else:
            preds = np.append(preds, logits, axis=0)
            out_label_ids = np.append(out_label_ids, label_ids, axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)

    print("***** Writting Predictions ******")
    logits0 = np.concatenate(predictions, axis=0)[:, 0]
    logits1 = np.concatenate(predictions, axis=0)[:, 1]
    score = pandas.DataFrame({'logits0': logits0, 'logits1': logits1, 'label': ground_truth})
    return score


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)


def rank_paras(data, pred_score):
    logits = np.array([pred_score['logits0'], pred_score['logits1']]).transpose()
    pred_score['prob'] = softmax(logits)[:, 1]

    ranked_paras = dict()
    cur_ptr = 0

    for case in tqdm(data):
        key = case['_id']
        tem_ptr = cur_ptr

        all_paras = []
        while cur_ptr < tem_ptr + len(case['context']):
            score = pred_score.loc[cur_ptr, 'prob'].item()
            all_paras.append((case['context'][cur_ptr - tem_ptr][0], score))
            cur_ptr += 1

        sorted_all_paras = sorted(all_paras, key=lambda x: x[1], reverse=True)
        ranked_paras[key] = sorted_all_paras

    return ranked_paras

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]

    label_list = processor.get_labels()
    examples = processor.get_examples(args.input_data)
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset

def set_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--eval_ckpt", default=None, type=str, required=True,
                        help="evaluation checkpoint")
    parser.add_argument("--raw_data", default=None, type=str, required=True,
                        help="raw data for processing")
    parser.add_argument("--input_data", default=None, type=str, required=True,
                        help="source data for processing")
    parser.add_argument("--data_dir", required=True, type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default='hotpotqa', type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = set_args()

    # Setup CUDA
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)

    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(args.eval_ckpt)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        config=config,
                                        state_dict=model_state_dict)
    model.cuda()
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)

    score = evaluate(args, model, tokenizer, prefix="")

    # load source data
    source_data = json.load(open(args.raw_data, 'r'))
    rank_paras_dict = rank_paras(source_data, score)
    json.dump(rank_paras_dict, open(join(args.data_dir, 'para_ranking.json'), 'w'))
