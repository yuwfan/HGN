from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import gzip
import pickle
import copy
import re
import os
import numpy as np
import itertools
import spacy

from spacy.tokenizer import Tokenizer
from collections import Counter
from tqdm import tqdm

from model_envs import MODEL_CLASSES
from envs import DATASET_FOLDER
from csr_mhqa.data_processing import Example, InputFeatures, get_cached_filename
from eval.hotpot_evaluate_v1 import normalize_answer

infix_re = re.compile(r'''[-—–~]''')
def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)
nlp = spacy.load("en_core_web_lg", disable=['tagger', 'parser'])
nlp.tokenizer.infix_finditer = infix_re.finditer
#nlp.tokenizer = custom_tokenizer(nlp)

def read_hotpot_examples(para_file,
                         full_file,
                         ner_file,
                         doc_link_file):

    with open(para_file, 'r', encoding='utf-8') as reader:
        para_data = json.load(reader)

    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    with open(ner_file, 'r', encoding='utf-8') as reader:
        ner_data = json.load(reader)

    with open(doc_link_file, 'r', encoding='utf-8') as reader:
        doc_link_data = json.load(reader)

    def split_sent(sent, offset=0):
        nlp_doc = nlp(sent)
        words, word_start_idx, char_to_word_offset = [], [], []
        for token in nlp_doc:
            # token match a-b, then split further
            words.append(token.text)
            word_start_idx.append(token.idx)

        word_offset = 0
        for c in range(len(sent)):
            if word_offset >= len(word_start_idx)-1 or c < word_start_idx[word_offset+1]:
                char_to_word_offset.append(word_offset + offset)
            else:
                char_to_word_offset.append(word_offset + offset + 1)
                word_offset += 1
        return words, char_to_word_offset, word_start_idx

    max_sent_cnt, max_entity_cnt = 0, 0

    examples = []
    for case in tqdm(full_data):
        key = case['_id']
        qas_type = case['type']
        sup_facts = set([(sp[0], sp[1])for sp in case['supporting_facts']])
        context = dict(case['context'])

        doc_tokens = []
        sent_names = []
        sup_facts_sent_id = []
        sup_para_id = set()
        sent_start_end_position = []
        para_start_end_position = []
        ques_entity_start_end_position = []
        ques_entities_text = []
        ctx_entity_start_end_position = []
        ctx_entities_text = []
        ctx_text = ""
        ans_start_position, ans_end_position = [], []
        ques_answer_ids, ctx_answer_ids = [], []

        title_to_id, title_id = {}, 0
        sent_to_id, sent_id = {}, 0
        s_e_edges = []
        s_s_edges = []
        p_s_edges = []

        ctx_answer_candidates = []
        ctx_char_to_word_offset = []  # Accumulated along all sentences
        ctx_word_to_char_idx = []

        # process question entity span
        question_text = case['question']
        question_tokens, ques_char_to_word_offset, ques_word_to_char_idx = split_sent(question_text)
        answer_norm = normalize_answer(case['answer'])

        q_e_edges = []
        for q_ent, q_start, q_end, q_type in ner_data[key]['question']:
            q_ent_text = question_text[q_start:q_end]
            if q_type != 'CONTEXT' and q_ent_text not in ques_entities_text:
                if len(ques_answer_ids) == 0 and normalize_answer(q_ent_text) == answer_norm:
                    ques_answer_ids.append(len(ques_entities_text))

                ques_entities_text.append(q_ent_text)
                q_e_edges.append((0, len(ques_entity_start_end_position))) # Q -> P; the id of Q is 0
                ques_entity_start_end_position.append((ques_char_to_word_offset[q_start], ques_char_to_word_offset[q_end-1]))

        sel_paras = para_data[key]
        ner_context = dict(ner_data[key]['context'])

        for title in itertools.chain.from_iterable(sel_paras):
            stripped_title= re.sub(r' \(.*?\)$', '', title)
            stripped_title_norm = normalize_answer(stripped_title)

            sents = context[title]
            sents_ner = ner_context[title]
            assert len(sents) == len(sents_ner)

            title_to_id[title] = title_id

            para_start_position = len(doc_tokens)
            prev_sent_id = None

            ctx_answer_set = set()
            for local_sent_id, (sent, sent_ner) in enumerate(zip(sents, sents_ner)):
                # Determine the global sent id for supporting facts
                local_sent_name = (title, local_sent_id)
                sent_to_id[local_sent_name] = sent_id
                sent_names.append(local_sent_name)

                # P -> S
                p_s_edges.append((title_id, sent_id))
                if prev_sent_id is not None:
                    # S -> S
                    s_s_edges.append((prev_sent_id, sent_id))

                sent += " "
                ctx_text += sent
                sent_start_word_id = len(doc_tokens)
                sent_start_char_id = len(ctx_char_to_word_offset)

                prev_is_whitespace = True
                cur_sent_words, cur_sent_char_to_word_offset, cur_sent_words_start_idx = split_sent(sent, offset=len(doc_tokens))
                doc_tokens.extend(cur_sent_words)
                ctx_char_to_word_offset.extend(cur_sent_char_to_word_offset)
                for cur_sent_word in cur_sent_words_start_idx:
                    ctx_word_to_char_idx.append(sent_start_char_id + cur_sent_word)
                assert len(doc_tokens) == len(ctx_word_to_char_idx)
                
                sent_start_end_position.append((sent_start_word_id, len(doc_tokens)-1))

                for sent_ner_id, (_, ent_start_char, ent_end_char, _) in enumerate(sent_ner):
                    if (ent_start_char, ent_end_char) in ctx_answer_set:
                        continue
                    s_ent_text = sent[ent_start_char:ent_end_char]
                    s_ent_text_norm = normalize_answer(s_ent_text)

                    if s_ent_text_norm == stripped_title_norm:
                        ctx_answer_candidates.append(len(ctx_entities_text))

                        if local_sent_name in sup_facts:
                            if len(ctx_answer_ids) == 0 and s_ent_text_norm == answer_norm:
                                ctx_answer_ids.append(len(ctx_entities_text))

                    ctx_entities_text.append(s_ent_text)
                    s_e_edges.append((sent_id, len(ctx_entity_start_end_position)))
                    ctx_entity_start_end_position.append((ctx_char_to_word_offset[sent_start_char_id + ent_start_char],
                                                          ctx_char_to_word_offset[sent_start_char_id + ent_end_char - 1]))

                    ctx_answer_set.add((ent_start_char, ent_end_char))

                # Find answer position
                if local_sent_name in sup_facts:
                    sup_para_id.add(title_id)
                    sup_facts_sent_id.append(sent_id)

                    answer_offsets = []
                    # find word offset
                    for cur_word_start_idx in cur_sent_words_start_idx:
                        if sent[cur_word_start_idx:cur_word_start_idx+len(case['answer'])] == case['answer']:
                            answer_offsets.append(cur_word_start_idx)
                    if len(answer_offsets) == 0:
                        answer_offset = sent.find(case['answer'])
                        if answer_offset != -1:
                            answer_offsets.append(answer_offset)
                    if case['answer'] not in ['yes', 'no'] and len(answer_offsets) > 0:
                        for answer_offset in answer_offsets:
                            start_char_position = sent_start_char_id + answer_offset
                            end_char_position = start_char_position + len(case['answer']) - 1
                            ans_start_position.append(ctx_char_to_word_offset[start_char_position])
                            ans_end_position.append(ctx_char_to_word_offset[end_char_position])
                prev_sent_id = sent_id
                sent_id += 1
            para_end_position = len(doc_tokens) - 1
            para_start_end_position.append((para_start_position, para_end_position, title))

            title_id += 1

        p_p_edges = []
        s_p_edges = []
        for _l in sel_paras[0]:
            for _r in sel_paras[1]:
                # edges: P -> P
                p_p_edges.append((title_to_id[_l], title_to_id[_r]))

                # edges: S -> P
                for local_sent_id, link_titles in enumerate(doc_link_data[_l]['hyperlink_titles']):
                    inter_titles = set(link_titles) & set(title_to_id.keys())
                    if len(inter_titles) > 0 and _r in inter_titles:
                        s_p_edges.append((sent_to_id[(_l, local_sent_id)], title_to_id[_r]))
        q_p_edges = [(0, title_to_id[para]) for para in sel_paras[0]]


        edges = {'ques_para': q_p_edges,
                 'para_para': p_p_edges,
                 'sent_sent': s_s_edges,
                 'para_sent': p_s_edges,
                 'sent_para': s_p_edges,
                 'ques_ent': q_e_edges,
                 'sent_ent': s_e_edges}

        max_sent_cnt = max(max_sent_cnt, len(sent_start_end_position))
        max_entity_cnt = max(max_entity_cnt, len(ctx_entity_start_end_position))

        if len(ans_start_position) > 1:
            # take the exact match for answer to avoid case of partial match
            start_position, end_position = [], []
            for _start_pos, _end_pos in zip(ans_start_position, ans_end_position):
                if normalize_answer(" ".join(doc_tokens[_start_pos:_end_pos+1])) == normalize_answer(case['answer']):
                    start_position.append(_start_pos) 
                    end_position.append(_end_pos) 
        else:
            start_position = ans_start_position
            end_position = ans_end_position
        
        example = Example(
            qas_id=key,
            qas_type=qas_type,
            question_tokens=question_tokens,
            doc_tokens=doc_tokens,
            sent_num=sent_id + 1,
            sent_names=sent_names,
            sup_fact_id=sup_facts_sent_id,
            sup_para_id=list(sup_para_id),
            ques_entities_text=ques_entities_text,
            ctx_entities_text=ctx_entities_text,
            para_start_end_position=para_start_end_position,
            sent_start_end_position=sent_start_end_position,
            ques_entity_start_end_position=ques_entity_start_end_position,
            ctx_entity_start_end_position=ctx_entity_start_end_position,
            question_text=case['question'],
            question_word_to_char_idx=ques_word_to_char_idx,
            ctx_text=ctx_text,
            ctx_word_to_char_idx=ctx_word_to_char_idx,
            edges=edges,
            orig_answer_text=case['answer'],
            answer_in_ques_entity_ids=ques_answer_ids,
            answer_in_ctx_entity_ids=ctx_answer_ids,
            answer_candidates_in_ctx_entity_ids=ctx_answer_candidates,
            start_position=start_position,
            end_position=end_position)
        examples.append(example)

    print("Maximum sentence cnt: {}".format(max_sent_cnt))
    print("Maximum entity cnt: {}".format(max_entity_cnt))

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length, max_entity_num,
                                 cls_token='[CLS]',
                                 sep_token='[SEP]',
                                 is_roberta=False,
                                 filter_no_ans=False):
    features = []
    failed = 0
    for (example_index, example) in enumerate(tqdm(examples)):
        def relocate_tok_span(orig_to_tok_index, orig_to_tok_back_index, word_tokens, subword_tokens, orig_start_position, orig_end_position, orig_text):
            if orig_start_position is None:
                return 0, 0

            tok_start_position = orig_to_tok_index[orig_start_position]
            if tok_start_position >= len(subword_tokens):
                return 0, 0

            if orig_end_position < len(word_tokens) - 1:
                tok_end_position = orig_to_tok_index[orig_end_position + 1] - 1
            else:
                tok_end_position = len(subword_tokens) - 1
            # Make answer span more accurate.
            if is_roberta: # hack for roberta now
                tok_end_position = orig_to_tok_back_index[orig_end_position]
                return tok_start_position, tok_end_position
            else:
                return _improve_answer_span(
                    subword_tokens, tok_start_position, tok_end_position, tokenizer, orig_text)

        all_query_tokens = [cls_token]
        tok_to_orig_index = [-1]
        ques_tok_to_orig_index = [0]
        ques_orig_to_tok_index = []
        ques_orig_to_tok_back_index = []

        for (i, token) in enumerate(example.question_tokens):
            ques_orig_to_tok_index.append(len(all_query_tokens))
            if is_roberta:
                sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
            else:
                sub_tokens = tokenizer.tokenize(token)

            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                ques_tok_to_orig_index.append(i)
                all_query_tokens.append(sub_token)
            ques_orig_to_tok_back_index.append(len(all_query_tokens) - 1)

        if is_roberta:
            all_query_tokens = all_query_tokens[:max_query_length-2]
            tok_to_orig_index = tok_to_orig_index[:max_query_length-2] + [-1, -1]
            # roberta uses an extra separator b/w pairs of sentences
            all_query_tokens += [sep_token, sep_token]
        else:
            all_query_tokens = all_query_tokens[:max_query_length-1]
            tok_to_orig_index = tok_to_orig_index[:max_query_length-1] + [-1]
            all_query_tokens += [sep_token]

        entity_spans = []
        answer_candidates_ids = []
        for q_ent_text, (q_ent_start, q_ent_end) in zip(example.ques_entities_text, example.ques_entity_start_end_position):
            _start_pos, _end_pos = relocate_tok_span(ques_orig_to_tok_index, ques_orig_to_tok_back_index, example.question_tokens, all_query_tokens, q_ent_start, q_ent_end, q_ent_text)
            if _start_pos < max_query_length and _end_pos < max_query_length:
                entity_spans.append((_start_pos, _end_pos))

        para_spans = []
        ques_entity_spans = []
        para_entity_spans = []
        sentence_spans = []
        all_doc_tokens = []
        orig_to_tok_index = []
        orig_to_tok_back_index = []

        all_doc_tokens += all_query_tokens

        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i+len(example.question_tokens))
                all_doc_tokens.append(sub_token)
            orig_to_tok_back_index.append(len(all_doc_tokens) - 1)

        for sent_span in example.sent_start_end_position:
            if sent_span[0] >= len(orig_to_tok_index) or sent_span[0] >= sent_span[1]:
                continue
            sent_start_position = orig_to_tok_index[sent_span[0]]
            sent_end_position = orig_to_tok_back_index[sent_span[1]]
            sentence_spans.append((sent_start_position, sent_end_position))

        for para_span in example.para_start_end_position:
            if para_span[0] >= len(orig_to_tok_index) or para_span[0] >= para_span[1]:
                continue
            para_start_position = orig_to_tok_index[para_span[0]]
            para_end_position = orig_to_tok_back_index[para_span[1]]
            para_spans.append((para_start_position, para_end_position, para_span[2]))

        # entity part
        q_entity_cnt = len(entity_spans)
        answer_candidates_ids.extend(list(range(q_entity_cnt))) # all entities in question would be in the answer candidates
        for c_ent_text, (c_ent_start, c_ent_end) in zip(example.ctx_entities_text, example.ctx_entity_start_end_position):
            _start_pos, _end_pos = relocate_tok_span(orig_to_tok_index, orig_to_tok_back_index, example.doc_tokens, all_doc_tokens, c_ent_start, c_ent_end, c_ent_text)
            if _start_pos < max_seq_length and _end_pos < max_seq_length:
                entity_spans.append((_start_pos, _end_pos))
        entity_spans = entity_spans[:max_entity_num]

        sent_max_index = _largest_valid_index(sentence_spans, max_seq_length-1)

        if sent_max_index < len(sentence_spans):
            sentence_spans = sentence_spans[:sent_max_index]
            max_tok_length = sentence_spans[-1][1]

            para_max_index = _largest_valid_index(para_spans, max_tok_length)
            max_para_span = para_spans[para_max_index]
            para_spans = para_spans[:para_max_index]
            para_spans.append((max_para_span[0], max_tok_length, max_para_span[2]))

            all_doc_tokens = all_doc_tokens[:max_tok_length]

            max_sent_cnt, max_para_cnt = len(sentence_spans), len(para_spans)

            max_entity_cnt = len(entity_spans)
            for _i in range(len(entity_spans)):
                if entity_spans[_i][1] >= max_tok_length:
                    max_entity_cnt = _i
                    break
            entity_spans = entity_spans[:max_entity_cnt]

            # deal with edges for out of bound
            def is_out_of_bound(keys, vals):
                key_list = keys.split('_')
                res = False
                for key, val in zip(key_list, vals):
                    if key == 'para' and val >= max_para_cnt:
                        res = True
                    if key == 'sent' and val >= max_sent_cnt:
                        res = True
                    if key == 'ent' and val >= max_entity_cnt:
                        res = True
                return res
            edges = {}
            for keys, edge in example.edges.items():
                if keys not in edges:
                    edges[keys] = []
                for _e in edge:
                    if not is_out_of_bound(keys, _e):
                        edges[keys].append(_e)

        else:
            edges = example.edges
        # reassign entity id
        sent_ent_edges = []
        for _l, _r in edges['sent_ent']:
            if _l < len(sentence_spans) and  _r < len(entity_spans):
                sent_ent_edges.append((_l, _r + q_entity_cnt))
        edges['sent_ent'] = sent_ent_edges

        # answer part
        ans_start_position, ans_end_position = [], []
        for ans_start, ans_end in zip(example.start_position, example.end_position):
            _start_pos, _end_pos = relocate_tok_span(orig_to_tok_index, orig_to_tok_back_index, example.doc_tokens, all_doc_tokens,
                                                     ans_start, ans_end, example.orig_answer_text)
            assert _start_pos <= _end_pos
            if _start_pos < len(all_doc_tokens) and _end_pos < len(all_doc_tokens):
                ans_start_position.append(_start_pos)
                ans_end_position.append(_end_pos)

        # answer entity
        c_entity_cnt = len(entity_spans) - q_entity_cnt

        answer_in_entity_ids = []
        for q_entity_id in example.answer_in_ques_entity_ids:
            if q_entity_id < q_entity_cnt:
                answer_in_entity_ids.append(q_entity_id)

        # entities which match title can be in the candidates
        for c_entity_id in example.answer_candidates_in_ctx_entity_ids:
            if c_entity_id < c_entity_cnt:
                answer_candidates_ids.append(c_entity_id + q_entity_cnt)

        if len(answer_in_entity_ids) == 0:
            for c_entity_id in example.answer_in_ctx_entity_ids:
                if c_entity_id < c_entity_cnt:
                    answer_in_entity_ids.append(c_entity_id + q_entity_cnt)
                    break

        if len(answer_in_entity_ids) > 0:
            assert answer_in_entity_ids[0] in answer_candidates_ids

        # Padding Document
        all_doc_tokens = all_doc_tokens[:max_seq_length - 1] + [sep_token]
        doc_input_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)
        query_input_ids = tokenizer.convert_tokens_to_ids(all_query_tokens)

        doc_input_mask = [1] * len(doc_input_ids)
        doc_segment_ids = [0] * len(query_input_ids) + [1] * (len(doc_input_ids) - len(query_input_ids))

        doc_pad_length = max_seq_length - len(doc_input_ids)
        doc_input_ids += [0] * doc_pad_length
        doc_input_mask += [0] * doc_pad_length
        doc_segment_ids += [0] * doc_pad_length

        # Padding Question
        query_input_mask = [1] * len(query_input_ids)
        query_segment_ids = [0] * len(query_input_ids)

        query_pad_length = max_query_length - len(query_input_ids)
        query_input_ids += [0] * query_pad_length
        query_input_mask += [0] * query_pad_length
        query_segment_ids += [0] * query_pad_length

        assert len(doc_input_ids) == max_seq_length
        assert len(doc_input_mask) == max_seq_length
        assert len(doc_segment_ids) == max_seq_length
        assert len(query_input_ids) == max_query_length
        assert len(query_input_mask) == max_query_length
        assert len(query_segment_ids) == max_query_length

        # Dropout out-of-bound span
        sup_fact_ids = [sent_id for sent_id in example.sup_fact_id if sent_id < len(sentence_spans)]
        sup_para_ids = [para_id for para_id in example.sup_para_id if para_id < len(para_spans)]
        if len(sup_fact_ids) != len(example.sup_fact_id):
            failed += 1

        if example.orig_answer_text == 'yes':
            ans_type = 1
        elif example.orig_answer_text == 'no':
            ans_type = 2
        elif len(answer_in_entity_ids) > 0:
            ans_type = 3
        else:
            ans_type = 0

        if filter_no_ans and ans_start_position[0] == 0 and ans_end_position[0] == 0:
            continue

        features.append(
            InputFeatures(qas_id=example.qas_id,
                          doc_tokens=all_doc_tokens,
                          doc_input_ids=doc_input_ids,
                          doc_input_mask=doc_input_mask,
                          doc_segment_ids=doc_segment_ids,
                          query_tokens=all_query_tokens,
                          query_input_ids=query_input_ids,
                          query_input_mask=query_input_mask,
                          query_segment_ids=query_segment_ids,
                          para_spans=para_spans,
                          sent_spans=sentence_spans,
                          entity_spans=entity_spans,
                          q_entity_cnt=q_entity_cnt,
                          sup_fact_ids=sup_fact_ids,
                          sup_para_ids=sup_para_ids,
                          ans_type=ans_type,
                          token_to_orig_map=tok_to_orig_index,
                          edges=edges,
                          orig_answer_text=example.orig_answer_text,
                          answer_in_entity_ids=answer_in_entity_ids,
                          answer_candidates_ids=answer_candidates_ids,
                          start_position=ans_start_position,
                          end_position=ans_end_position)
        )
    print(failed)

    return features


def _largest_valid_index(spans, limit):
    for idx in range(len(spans)):
        if spans[idx][1] >= limit:
            return idx
    return len(spans)

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def create_graphs(case, max_para_num, max_sent_num, max_entity_num):
    max_node_cnt = max_para_num + max_sent_num + max_entity_num + 1 # ques + para + sent
    adj = np.zeros((max_node_cnt, max_node_cnt), dtype=np.float32)

    def get_id(key, idx):
        # ques: 0
        # para: 1 + #para_id
        # sent: 1 + max_para_num + #sent_id
        # entity: 1 + max_para_num + max_sent_num + #entity_id
        if key == 'ques':
            return 0
        elif key == 'para':
            return 1 + idx if idx < max_para_num else -1
        elif key == 'sent':
            return 1 + max_para_num + idx if idx < max_sent_num else -1
        elif key == 'ent':
            return 1 + max_para_num + max_sent_num + idx if idx < max_entity_num else -1
        else:
            raise ValueError("{} is not supported.")

    edge_types = ['ques_para', 'ques_ent', 'para_para', 'para_sent', 'sent_para', 'sent_sent', 'sent_ent']
    for idx, key in enumerate(edge_types):
        edge = case.edges[key]
        key_l, key_r = key.split('_')
        for _l, _r in edge:
            new_l, new_r = get_id(key_l, _l), get_id(key_r, _r)
            if new_l != -1 and new_r != -1:
                adj[new_l, new_r] = adj[new_r, new_l] = idx + 1

    return adj

def build_graph(args, examples, features, entity_num):
    examples_dict = {e.qas_id: e for e in examples}

    graphs = {}
    for case in tqdm(features):
        graph = create_graphs(case,
                              max_para_num=4,
                              max_sent_num=args.max_sent_num,
                              max_entity_num=entity_num)
        graphs[case.qas_id] = {'adj': graph}

    return graphs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--para_path", type=str, required=True)
    parser.add_argument("--full_data", type=str, required=True)
    parser.add_argument("--ner_path", type=str, required=True)
    parser.add_argument("--doc_link_ner", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help='define output directory')
    parser.add_argument("--graph_id", type=str, default="1", help='define output directory')

    # Other parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_entity_num", default=60, type=int)
    parser.add_argument("--max_sent_num", default=40, type=int)
    parser.add_argument("--max_query_length", default=50, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--filter_no_ans", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    args = parser.parse_args()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    examples = read_hotpot_examples(para_file=args.para_path,
                                    full_file=args.full_data,
                                    ner_file=args.ner_path,
                                    doc_link_file=args.doc_link_ner)
    cached_examples_file = os.path.join(args.output_dir,
                                        get_cached_filename('examples', args))
    with gzip.open(cached_examples_file, 'wb') as fout:
        pickle.dump(examples, fout)

    features = convert_examples_to_features(examples, tokenizer,
                                            max_seq_length=args.max_seq_length,
                                            max_query_length=args.max_query_length,
                                            max_entity_num=args.max_entity_num,
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            is_roberta=bool(args.model_type in ['roberta']),
                                            filter_no_ans=args.filter_no_ans)
    cached_features_file = os.path.join(args.output_dir,
                                        get_cached_filename('features',  args))

    with gzip.open(cached_features_file, 'wb') as fout:
        pickle.dump(features, fout)

    # build graphs
    cached_graph_file = os.path.join(args.output_dir,
                                     get_cached_filename('graphs', args))

    graphs = build_graph(args, examples, features, args.max_entity_num)
    with gzip.open(cached_graph_file, 'wb') as fout:
        pickle.dump(graphs, fout)
