import gzip
import pickle
import json
import torch
import numpy as np
import os

from os.path import join
from tqdm import tqdm
from numpy.random import shuffle

from envs import DATASET_FOLDER

IGNORE_INDEX = -100

def get_cached_filename(f_type, config):
    assert f_type in ['examples', 'features', 'graphs']

    return f"cached_{f_type}_{config.model_type}_{config.max_seq_length}_{config.max_query_length}.pkl.gz"

class Example(object):

    def __init__(self,
                 qas_id,
                 qas_type,
                 question_tokens,
                 doc_tokens,
                 sent_num,
                 sent_names,
                 sup_fact_id,
                 sup_para_id,
                 ques_entities_text,
                 ctx_entities_text,
                 para_start_end_position,
                 sent_start_end_position,
                 ques_entity_start_end_position,
                 ctx_entity_start_end_position,
                 question_text,
                 question_word_to_char_idx,
                 ctx_text,
                 ctx_word_to_char_idx,
                 edges=None,
                 orig_answer_text=None,
                 answer_in_ques_entity_ids=None,
                 answer_in_ctx_entity_ids=None,
                 answer_candidates_in_ctx_entity_ids=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.sup_fact_id = sup_fact_id
        self.sup_para_id = sup_para_id
        self.ques_entities_text = ques_entities_text
        self.ctx_entities_text = ctx_entities_text
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.ques_entity_start_end_position = ques_entity_start_end_position
        self.ctx_entity_start_end_position = ctx_entity_start_end_position
        self.question_word_to_char_idx = question_word_to_char_idx
        self.ctx_text = ctx_text
        self.ctx_word_to_char_idx = ctx_word_to_char_idx
        self.edges = edges
        self.orig_answer_text = orig_answer_text
        self.answer_in_ques_entity_ids = answer_in_ques_entity_ids
        self.answer_in_ctx_entity_ids = answer_in_ctx_entity_ids
        self.answer_candidates_in_ctx_entity_ids= answer_candidates_in_ctx_entity_ids
        self.start_position = start_position
        self.end_position = end_position

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,
                 query_tokens,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 para_spans,
                 sent_spans,
                 entity_spans,
                 q_entity_cnt,
                 sup_fact_ids,
                 sup_para_ids,
                 ans_type,
                 token_to_orig_map,
                 edges=None,
                 orig_answer_text=None,
                 answer_in_entity_ids=None,
                 answer_candidates_ids=None,
                 start_position=None,
                 end_position=None):

        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_input_ids = doc_input_ids
        self.doc_input_mask = doc_input_mask
        self.doc_segment_ids = doc_segment_ids

        self.query_tokens = query_tokens
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids

        self.para_spans = para_spans
        self.sent_spans = sent_spans
        self.entity_spans = entity_spans
        self.q_entity_cnt = q_entity_cnt
        self.sup_fact_ids = sup_fact_ids
        self.sup_para_ids = sup_para_ids
        self.ans_type = ans_type

        self.edges = edges
        self.token_to_orig_map = token_to_orig_map
        self.orig_answer_text = orig_answer_text
        self.answer_in_entity_ids = answer_in_entity_ids
        self.answer_candidates_ids = answer_candidates_ids

        self.start_position = start_position
        self.end_position = end_position


class DataIteratorPack(object):
    def __init__(self,
                 features, example_dict, graph_dict,
                 bsz, device,
                 para_limit, sent_limit, ent_limit, ans_ent_limit,
                 mask_edge_types,
                 sequential=False):
        self.bsz = bsz
        self.device = device
        self.features = features
        self.example_dict = example_dict
        self.graph_dict = graph_dict
        self.sequential = sequential
        self.para_limit = para_limit
        self.sent_limit = sent_limit
        self.ent_limit = ent_limit
        self.ans_ent_limit = ans_ent_limit
        self.graph_nodes_num = 1 + para_limit + sent_limit + ent_limit
        self.example_ptr = 0
        self.mask_edge_types = mask_edge_types
        self.max_seq_length = 512
        if not sequential:
            shuffle(self.features)

    def refresh(self):
        self.example_ptr = 0
        if not self.sequential:
            shuffle(self.features)

    def empty(self):
        return self.example_ptr >= len(self.features)

    def __len__(self):
        return int(np.ceil(len(self.features)/self.bsz))

    def __iter__(self):
        # BERT input
        context_idxs = torch.LongTensor(self.bsz, self.max_seq_length)
        context_mask = torch.LongTensor(self.bsz, self.max_seq_length)
        segment_idxs = torch.LongTensor(self.bsz, self.max_seq_length)

        # Mappings
        query_mapping = torch.Tensor(self.bsz, self.max_seq_length).cuda(self.device)
        para_start_mapping = torch.Tensor(self.bsz, self.para_limit, self.max_seq_length).cuda(self.device)
        para_end_mapping = torch.Tensor(self.bsz, self.para_limit, self.max_seq_length).cuda(self.device)
        para_mapping = torch.Tensor(self.bsz, self.max_seq_length, self.para_limit).cuda(self.device)
        sent_start_mapping = torch.Tensor(self.bsz, self.sent_limit, self.max_seq_length).cuda(self.device)
        sent_end_mapping = torch.Tensor(self.bsz, self.sent_limit, self.max_seq_length).cuda(self.device)
        sent_mapping = torch.Tensor(self.bsz, self.max_seq_length, self.sent_limit).cuda(self.device)
        ent_start_mapping = torch.Tensor(self.bsz, self.ent_limit, self.max_seq_length).cuda(self.device)
        ent_end_mapping = torch.Tensor(self.bsz, self.ent_limit, self.max_seq_length).cuda(self.device)
        ent_mapping = torch.Tensor(self.bsz, self.max_seq_length, self.ent_limit).cuda(self.device)

        # Mask
        para_mask = torch.FloatTensor(self.bsz, self.para_limit).cuda(self.device)
        sent_mask = torch.FloatTensor(self.bsz, self.sent_limit).cuda(self.device)
        ent_mask = torch.FloatTensor(self.bsz, self.ent_limit).cuda(self.device)
        ans_cand_mask = torch.FloatTensor(self.bsz, self.ent_limit).cuda(self.device)

        # Label tensor
        y1 = torch.LongTensor(self.bsz).cuda(self.device)
        y2 = torch.LongTensor(self.bsz).cuda(self.device)
        q_type = torch.LongTensor(self.bsz).cuda(self.device)
        is_support = torch.FloatTensor(self.bsz, self.sent_limit).cuda(self.device)
        is_gold_para = torch.FloatTensor(self.bsz, self.para_limit).cuda(self.device)
        is_gold_ent = torch.FloatTensor(self.bsz).cuda(self.device)

        # Graph related
        graphs = torch.Tensor(self.bsz, self.graph_nodes_num, self.graph_nodes_num).cuda(self.device)

        while True:
            if self.example_ptr >= len(self.features):
                break
            start_id = self.example_ptr
            cur_bsz = min(self.bsz, len(self.features) - start_id)
            cur_batch = self.features[start_id: start_id + cur_bsz]
            cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)

            ids = []
            for mapping in [para_mapping, para_start_mapping, para_end_mapping, \
                            sent_mapping, sent_start_mapping, sent_end_mapping, \
                            ent_mapping, ent_start_mapping, ent_end_mapping, \
                            ans_cand_mask,
                            query_mapping]:
                mapping.zero_()

            is_support.fill_(IGNORE_INDEX)
            is_gold_para.fill_(IGNORE_INDEX)
            is_gold_ent.fill_(IGNORE_INDEX)

            for i in range(len(cur_batch)):
                case = cur_batch[i]
                context_idxs[i].copy_(torch.Tensor(case.doc_input_ids))
                context_mask[i].copy_(torch.Tensor(case.doc_input_mask))
                segment_idxs[i].copy_(torch.Tensor(case.doc_segment_ids))

                if len(case.sent_spans) > 0:
                    for j in range(case.sent_spans[0][0] - 1):
                        query_mapping[i, j] = 1

                for j, para_span in enumerate(case.para_spans[:self.para_limit]):
                    is_gold_flag = j in case.sup_para_ids
                    start, end, _ = para_span
                    if start <= end:
                        end = min(end, self.max_seq_length-1)
                        is_gold_para[i, j] = int(is_gold_flag)
                        para_mapping[i, start:end+1, j] = 1
                        para_start_mapping[i, j, start] = 1
                        para_end_mapping[i, j, end] = 1

                for j, sent_span in enumerate(case.sent_spans[:self.sent_limit]):
                    is_sp_flag = j in case.sup_fact_ids
                    start, end = sent_span
                    if start <= end:
                        end = min(end, self.max_seq_length-1)
                        is_support[i, j] = int(is_sp_flag)
                        sent_mapping[i, start:end+1, j] = 1
                        sent_start_mapping[i, j, start] = 1
                        sent_end_mapping[i, j, end] = 1

                for j, ent_span in enumerate(case.entity_spans[:self.ent_limit]):
                    start, end = ent_span
                    if start <= end:
                        end = min(end, self.max_seq_length-1)
                        ent_mapping[i, start:end+1, j] = 1
                        ent_start_mapping[i, j, start] = 1
                        ent_end_mapping[i, j, end] = 1
                    ans_cand_mask[i, j] = int(j in case.answer_candidates_ids)

                is_gold_ent[i] = case.answer_in_entity_ids[0] if len(case.answer_in_entity_ids) > 0 else IGNORE_INDEX

                if case.ans_type == 0 or case.ans_type == 3:
                    if len(case.end_position) == 0:
                        y1[i] = y2[i] = 0
                    elif case.end_position[0] < self.max_seq_length and context_mask[i][case.end_position[0]+1] == 1: # "[SEP]" is the last token
                        y1[i] = case.start_position[0]
                        y2[i] = case.end_position[0]
                    else:
                        y1[i] = y2[i] = 0
                    q_type[i] = case.ans_type if is_gold_ent[i] > 0 else 0
                elif case.ans_type == 1:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 1
                elif case.ans_type == 2:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 2
                # ignore entity loss if there is no entity
                if case.ans_type != 3:
                    is_gold_ent[i].fill_(IGNORE_INDEX)

                tmp_graph = self.graph_dict[case.qas_id]
                graph_adj = torch.from_numpy(tmp_graph['adj']).to(self.device)
                for k in range(graph_adj.size(0)):
                    graph_adj[k, k] = 8
                for edge_type in self.mask_edge_types:
                    graph_adj = torch.where(graph_adj == edge_type, torch.zeros_like(graph_adj), graph_adj)
                graphs[i] = graph_adj

                ids.append(case.qas_id)

            input_lengths = (context_mask[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            para_mask = (para_mapping > 0).any(1).float()
            sent_mask = (sent_mapping > 0).any(1).float()
            ent_mask = (ent_mapping > 0).any(1).float()

            self.example_ptr += cur_bsz

            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous().to(self.device),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous().to(self.device),
                'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous().to(self.device),
                'context_lens': input_lengths.contiguous().to(self.device),
                'y1': y1[:cur_bsz],
                'y2': y2[:cur_bsz],
                'ids': ids,
                'q_type': q_type[:cur_bsz],
                'is_support': is_support[:cur_bsz, :].contiguous(),
                'is_gold_para': is_gold_para[:cur_bsz, :].contiguous(),
                'is_gold_ent': is_gold_ent[:cur_bsz].contiguous(),
                'query_mapping': query_mapping[:cur_bsz, :max_c_len].contiguous(),
                'para_mapping': para_mapping[:cur_bsz, :max_c_len, :],
                'para_start_mapping': para_start_mapping[:cur_bsz, :, :max_c_len],
                'para_end_mapping': para_end_mapping[:cur_bsz, :, :max_c_len],
                'para_mask': para_mask[:cur_bsz, :],
                'sent_mapping': sent_mapping[:cur_bsz, :max_c_len, :],
                'sent_start_mapping': sent_start_mapping[:cur_bsz, :, :max_c_len],
                'sent_end_mapping': sent_end_mapping[:cur_bsz, :, :max_c_len],
                'sent_mask': sent_mask[:cur_bsz, :],
                'ent_mapping': ent_mapping[:cur_bsz, :max_c_len, :],
                'ent_start_mapping': ent_start_mapping[:cur_bsz, :, :max_c_len],
                'ent_end_mapping': ent_end_mapping[:cur_bsz, :, :max_c_len],
                'ent_mask': ent_mask[:cur_bsz, :],
                'ans_cand_mask': ans_cand_mask[:cur_bsz, :],
                'graphs': graphs[:cur_bsz, :, :]
            }

class DataHelper:
    def __init__(self, gz=True, config=None):
        self.DataIterator = DataIteratorPack
        self.gz = gz
        self.suffix = '.pkl.gz' if gz else '.pkl'

        self.data_dir = join(DATASET_FOLDER, 'data_feat')

        self.__train_features__ = None
        self.__dev_features__ = None

        self.__train_examples__ = None
        self.__dev_examples__ = None

        self.__train_graphs__ = None
        self.__dev_graphs__ = None

        self.__train_example_dict__ = None
        self.__dev_example_dict__ = None

        self.config = config

    def get_feature_file(self, tag):
        cached_filename = get_cached_filename('features', self.config)
        return join(self.data_dir, tag, cached_filename)

    def get_example_file(self, tag):
        cached_filename = get_cached_filename('examples', self.config)

        return join(self.data_dir, tag, cached_filename)

    def get_graph_file(self, tag):
        cached_filename = get_cached_filename('graphs', self.config)
        return join(self.data_dir, tag, cached_filename)

    @property
    def train_feature_file(self):
        return self.get_feature_file('train')

    @property
    def dev_feature_file(self):
        return self.get_feature_file('dev_distractor')

    @property
    def train_example_file(self):
        return self.get_example_file('train')

    @property
    def dev_example_file(self):
        return self.get_example_file('dev_distractor')

    @property
    def train_graph_file(self):
        return self.get_graph_file('train')

    @property
    def dev_graph_file(self):
        return self.get_graph_file('dev_distractor')

    def get_pickle_file(self, file_name):
        if self.gz:
            return gzip.open(file_name, 'rb')
        else:
            return open(file_name, 'rb')

    def __get_or_load__(self, name, file):
        if getattr(self, name) is None:
            with self.get_pickle_file(file) as fin:
                print('loading', file)
                setattr(self, name, pickle.load(fin))

        return getattr(self, name)

    # Features
    @property
    def train_features(self):
        return self.__get_or_load__('__train_features__', self.train_feature_file)

    @property
    def dev_features(self):
        return self.__get_or_load__('__dev_features__', self.dev_feature_file)

    # Examples
    @property
    def train_examples(self):
        return self.__get_or_load__('__train_examples__', self.train_example_file)

    @property
    def dev_examples(self):
        return self.__get_or_load__('__dev_examples__', self.dev_example_file)

    # Graphs
    @property
    def train_graphs(self):
        return self.__get_or_load__('__train_graphs__', self.train_graph_file)

    @property
    def dev_graphs(self):
        return self.__get_or_load__('__dev_graphs__', self.dev_graph_file)

    # Example dict
    @property
    def train_example_dict(self):
        if self.__train_example_dict__ is None:
            self.__train_example_dict__ = {e.qas_id: e for e in self.train_examples}
        return self.__train_example_dict__

    @property
    def dev_example_dict(self):
        if self.__dev_example_dict__ is None:
            self.__dev_example_dict__ = {e.qas_id: e for e in self.dev_examples}
        return self.__dev_example_dict__

    # Feature dict
    @property
    def train_feature_dict(self):
        return {e.qas_id: e for e in self.train_features}

    @property
    def dev_feature_dict(self):
        return {e.qas_id: e for e in self.dev_features}

    # Load
    def load_dev(self):
        return self.dev_features, self.dev_example_dict, self.dev_graphs

    def load_train(self):
        return self.train_features, self.train_example_dict, self.train_graphs

    @property
    def dev_loader(self):
        return self.DataIterator(*self.load_dev(),
                                 bsz=self.config.eval_batch_size,
                                 device=self.config.device,
                                 para_limit=self.config.max_para_num,
                                 sent_limit=self.config.max_sent_num,
                                 ent_limit=self.config.max_entity_num,
                                 ans_ent_limit=self.config.max_ans_ent_num,
                                 mask_edge_types=self.config.mask_edge_types,
                                 sequential=True)

    @property
    def train_loader(self):
        return self.DataIterator(*self.load_train(),
                                 bsz=self.config.batch_size,
                                 device=self.config.device,
                                 para_limit=self.config.max_para_num,
                                 sent_limit=self.config.max_sent_num,
                                 ent_limit=self.config.max_entity_num,
                                 ans_ent_limit=self.config.max_ans_ent_num,
                                 mask_edge_types=self.config.mask_edge_types,
                                 sequential=False)
