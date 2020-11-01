import pickle
import torch
import json
import numpy as np
import string
import re
import os
import shutil
import collections
import logging
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm

from model_envs import MODEL_CLASSES, ALL_MODELS
from transformers.tokenization_bert import whitespace_tokenize, BasicTokenizer, BertTokenizer
from transformers import AdamW
from eval.hotpot_evaluate_v1 import normalize_answer, eval as hotpot_eval
from csr_mhqa.data_processing import IGNORE_INDEX

logger = logging.getLogger(__name__)

def load_encoder_model(encoder_name_or_path, model_type):
    if encoder_name_or_path in [None, 'None', 'none']:
        raise ValueError('no checkpoint provided for model!')

    config_class, model_encoder, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(encoder_name_or_path)
    if config is None:
        raise ValueError(f'config.json is not found at {encoder_name_or_path}')

    # check if is a path
    if os.path.exists(encoder_name_or_path):
        if os.path.isfile(os.path.join(encoder_name_or_path, 'pytorch_model.bin')):
            encoder_file = os.path.join(encoder_name_or_path, 'pytorch_model.bin')
        else:
            encoder_file = os.path.join(encoder_name_or_path, 'encoder.pkl')
        encoder = model_encoder.from_pretrained(encoder_file, config=config)
    else:
        encoder = model_encoder.from_pretrained(encoder_name_or_path, config=config)

    return encoder, config


def get_optimizer(encoder, model, args, learning_rate, remove_pooler=False):
    """
    get BertAdam for encoder / classifier or BertModel
    :param model:
    :param classifier:
    :param args:
    :param remove_pooler:
    :return:
    """

    param_optimizer = list(encoder.named_parameters())
    param_optimizer += list(model.named_parameters())

    if remove_pooler:
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=args.adam_epsilon)

    return optimizer

def compute_loss(args, batch, start, end, para, sent, ent, q_type):
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    binary_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    loss_span = args.ans_lambda * (criterion(start, batch['y1']) + criterion(end, batch['y2']))
    loss_type = args.type_lambda * criterion(q_type, batch['q_type'])

    sent_pred = sent.view(-1, 2)
    sent_gold = batch['is_support'].long().view(-1)
    loss_sup = args.sent_lambda * criterion(sent_pred, sent_gold.long())

    loss_ent = args.ent_lambda * criterion(ent, batch['is_gold_ent'].long())
    loss_para = args.para_lambda * criterion(para.view(-1, 2), batch['is_gold_para'].long().view(-1))

    loss = loss_span + loss_type + loss_sup + loss_ent + loss_para

    return loss, loss_span, loss_type, loss_sup, loss_ent, loss_para


def eval_model(args, encoder, model, dataloader, example_dict, feature_dict, prediction_file, eval_file, dev_gold_file):
    encoder.eval()
    model.eval()

    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}

    dataloader.refresh()

    thresholds = np.arange(0.1, 1.0, 0.05)
    N_thresh = len(thresholds)
    total_sp_dict = [{} for _ in range(N_thresh)]

    for batch in tqdm(dataloader):
        with torch.no_grad():
            inputs = {'input_ids':      batch['context_idxs'],
                      'attention_mask': batch['context_mask'],
                      'token_type_ids': batch['segment_idxs'] if args.model_type in ['bert', 'xlnet'] else None}  # XLM don't use segment_ids
            outputs = encoder(**inputs)

            batch['context_encoding'] = outputs[0]
            batch['context_mask'] = batch['context_mask'].float().to(args.device)
            start, end, q_type, paras, sent, ent, yp1, yp2 = model(batch, return_yp=True)

        type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
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

    return best_metrics, best_threshold


def get_weights(size, gain=1.414):
    weights = nn.Parameter(torch.zeros(size=size))
    nn.init.xavier_uniform_(weights, gain=gain)
    return weights


def get_bias(size):
    bias = nn.Parameter(torch.zeros(size=size))
    return bias


def get_act(act):
    if act.startswith('lrelu'):
        return nn.LeakyReLU(float(act.split(':')[1]))
    elif act == 'relu':
        return nn.ReLU()
    else:
        raise NotImplementedError

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            print("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            print("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

def convert_to_tokens(examples, features, ids, y1, y2, q_type_prob):
    answer_dict, answer_type_dict = {}, {}
    answer_type_prob_dict = {}

    q_type = np.argmax(q_type_prob, 1)

    def get_ans_from_pos(qid, y1, y2):
        feature = features[qid]
        example = examples[qid]

        tok_to_orig_map = feature.token_to_orig_map
        orig_all_tokens = example.question_tokens + example.doc_tokens

        final_text = " "
        if y1 < len(tok_to_orig_map) and y2 < len(tok_to_orig_map):
            orig_tok_start = tok_to_orig_map[y1]
            orig_tok_end = tok_to_orig_map[y2]

            ques_tok_len = len(example.question_tokens)
            if orig_tok_start < ques_tok_len and orig_tok_end < ques_tok_len:
                ques_start_idx = example.question_word_to_char_idx[orig_tok_start]
                ques_end_idx = example.question_word_to_char_idx[orig_tok_end] + len(example.question_tokens[orig_tok_end])
                final_text = example.question_text[ques_start_idx:ques_end_idx]
            else:
                orig_tok_start -= len(example.question_tokens)
                orig_tok_end -= len(example.question_tokens)
                ctx_start_idx = example.ctx_word_to_char_idx[orig_tok_start]
                ctx_end_idx = example.ctx_word_to_char_idx[orig_tok_end] + len(example.doc_tokens[orig_tok_end])
                final_text = example.ctx_text[example.ctx_word_to_char_idx[orig_tok_start]:example.ctx_word_to_char_idx[orig_tok_end]+len(example.doc_tokens[orig_tok_end])]

        return final_text

    for i, qid in enumerate(ids):
        feature = features[qid]
        answer_text = ''
        if q_type[i] in [0, 3]:
            answer_text = get_ans_from_pos(qid, y1[i], y2[i])
        elif q_type[i] == 1:
            answer_text = 'yes'
        elif q_type[i] == 2:
            answer_text = 'no'
        else: 
            raise ValueError("question type error")

        answer_dict[qid] = answer_text
        answer_type_prob_dict[qid] = q_type_prob[i].tolist()
        answer_type_dict[qid] = q_type[i].item()

    return answer_dict, answer_type_dict, answer_type_prob_dict

def count_parameters(model, trainable_only=True, is_dict=False):
    """
    Count number of parameters in a model or state dictionary
    :param model:
    :param trainable_only:
    :param is_dict:
    :return:
    """
    if is_dict:
        return sum(np.prod(list(model[k].size())) for k in model)
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
