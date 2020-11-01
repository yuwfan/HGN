# coding=utf-8
#/usr/bin/env python3
import os
import argparse
import torch
import json
import logging
import random
import numpy as np

from os.path import join

from envs import DATASET_FOLDER, MODEL_FOLDER, OUTPUT_FOLDER
from model_envs import ALL_MODELS, MODEL_CLASSES

logger = logging.getLogger(__name__)


def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def json_to_argv(json_file):
    j = json.load(open(json_file))
    argv = []
    for k, v in j.items():
        new_v = str(v) if v is not None else None
        argv.extend(['--' + k, new_v])
    return argv

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def complete_default_train_parser(args):
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # set n_gpu
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.data_parallel:
            args.n_gpu = torch.cuda.device_count()
        else:
            args.n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    args.num_gnn_layers = int(args.gnn.split(':')[1].split(',')[0])
    args.num_gnn_heads = int(args.gnn.split(':')[1].split(',')[1])
    if len(args.mask_edge_types):
        args.mask_edge_types = list(map(int, args.mask_edge_types.split(',')))
    args.max_doc_len = 512
    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # TODO: only support albert-xxlarge-v2 now
    args.input_dim = 768 if 'base' in args.encoder_name_or_path else (4096 if 'albert' in args.encoder_name_or_path else 1024)

    # output dir name
    if not args.exp_name:
        args.exp_name = '_'.join([args.encoder_name_or_path,
                          'lr' + str(args.learning_rate),
                          'bs' + str(args.batch_size)])
    args.exp_name = os.path.join(args.output_dir, args.exp_name)

    set_seed(args)
    os.makedirs(args.exp_name, exist_ok=True)
    torch.save(args, join(args.exp_name, "training_args.bin"))

    return args


def default_train_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir',
                        type=str,
                        default=OUTPUT_FOLDER,
                        help='Directory to save model and summaries')
    parser.add_argument("--exp_name",
                        type=str,
                        default=None,
                        help="If set, this will be used as directory name in OUTOUT folder")
    parser.add_argument("--config_file",
                        type=str,
                        default=None,
                        help="configuration file for command parser")
    parser.add_argument("--dev_gold_file",
                        type=str,
                        default=join(DATASET_FOLDER, 'data_raw', 'hotpot_dev_distractor_v1.json'))

    # model
    parser.add_argument("--model_type",
                        default='bert',
                        type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--max_query_length", default=50, type=int)
    parser.add_argument("--encoder_name_or_path",
                        default='bert-base-uncased',
                        type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size",
                        default=8,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int)

    # eval
    parser.add_argument("--encoder_ckpt", default=None, type=str)
    parser.add_argument("--model_ckpt", default=None, type=str)

    # Environment
    parser.add_argument("--data_parallel",
                        default=False,
                        type=boolean_string,
                        help="use data parallel or not")
    parser.add_argument("--gpu_id", default=None, type=str, help="GPU id")
    parser.add_argument('--fp16',
                        type=boolean_string,
                        default='false',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    # learning and log
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")

    # hyper-parameter
    parser.add_argument('--q_update', type=boolean_string, default='False', help='Whether update query')
    parser.add_argument("--trans_drop", type=float, default=0.2)
    parser.add_argument("--trans_heads", type=int, default=3)

    # graph
    parser.add_argument('--num_edge_type', type=int, default=8)
    parser.add_argument('--mask_edge_types', type=str, default="0")

    parser.add_argument('--gnn', default='gat:1,2', type=str, help='gat:n_layer, n_head')
    parser.add_argument("--gnn_drop", type=float, default=0.3)
    parser.add_argument('--q_attn', type=boolean_string, default='True', help='whether use query attention in GAT')
    parser.add_argument("--lstm_drop", type=float, default=0.3)

    parser.add_argument("--max_para_num", default=4, type=int)
    parser.add_argument("--max_sent_num", default=40, type=int)
    parser.add_argument("--max_entity_num", default=60, type=int)
    parser.add_argument("--max_ans_ent_num", default=15, type=int)

    # bi attn
    parser.add_argument('--ctx_attn', type=str, default='gate_att_up', choices=['no_gate', 'gate_att_or', 'gate_att_up'])
    parser.add_argument("--ctx_attn_hidden_dim", type=int, default=300)
    parser.add_argument("--bi_attn_drop", type=float, default=0.3)
    parser.add_argument("--hidden_dim", type=int, default=300)

    # loss
    parser.add_argument("--ans_lambda", type=float, default=1)
    parser.add_argument("--type_lambda", type=float, default=1)
    parser.add_argument("--para_lambda", type=float, default=1)
    parser.add_argument("--sent_lambda", type=float, default=5)
    parser.add_argument("--ent_lambda", type=float, default=1)
    parser.add_argument("--sp_threshold", type=float, default=0.5)

    return parser
