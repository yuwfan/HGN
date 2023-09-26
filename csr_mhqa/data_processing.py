from os.path import join as os_path_join
from gzip import open as gzip_open
from pickle import load as pickle_load
from torch import (
    Tensor as torch_Tensor,
    LongTensor as torch_LongTensor,
    FloatTensor as torch_FloatTensor,
    zeros_like as torch_zeros_like,
    from_numpy as torch_from_numpy,
    where as torch_where,
)
from numpy import ceil as np_ceil
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
        return int(np_ceil(len(self.features)/self.bsz))

    def __iter__(self):
        # Cached values
        device = self.device
        bsz = self.bsz
        para_limit = self.para_limit
        graph_dict = self.graph_dict
        max_seq_length = self.max_seq_length
        sent_limit = self.sent_limit
        ent_limit = self.ent_limit

        # BERT input
        context_idxs = torch_LongTensor(bsz, max_seq_length)
        context_mask = torch_LongTensor(bsz, max_seq_length)
        segment_idxs = torch_LongTensor(bsz, max_seq_length)

        # Mappings
        query_mapping = torch_Tensor(bsz, max_seq_length, device=device)
        para_start_mapping = torch_Tensor(bsz, para_limit, max_seq_length, device=device)
        para_end_mapping = torch_Tensor(bsz, para_limit, max_seq_length, device=device)
        para_mapping = torch_Tensor(bsz, max_seq_length, para_limit, device=device)
        sent_start_mapping = torch_Tensor(bsz, sent_limit, max_seq_length, device=device)
        sent_end_mapping = torch_Tensor(bsz, sent_limit, max_seq_length, device=device)
        sent_mapping = torch_Tensor(bsz, max_seq_length, sent_limit, device=device)
        ent_start_mapping = torch_Tensor(bsz, ent_limit, max_seq_length, device=device)
        ent_end_mapping = torch_Tensor(bsz, ent_limit, max_seq_length, device=device)
        ent_mapping = torch_Tensor(bsz, max_seq_length, ent_limit, device=device)

        # Mask
        para_mask = torch_FloatTensor(bsz, para_limit, device=device)
        sent_mask = torch_FloatTensor(bsz, sent_limit, device=device)
        ent_mask = torch_FloatTensor(bsz, ent_limit, device=device)
        ans_cand_mask = torch_FloatTensor(bsz, ent_limit, device=device)

        # Label tensor
        y1 = torch_LongTensor(bsz, device=device)
        y2 = torch_LongTensor(bsz, device=device)
        q_type = torch_LongTensor(bsz, device=device)
        is_support = torch_FloatTensor(bsz, sent_limit, device=device)
        is_gold_para = torch_FloatTensor(bsz, para_limit, device=device)
        is_gold_ent = torch_FloatTensor(bsz, device=device)

        # Graph related
        graph_nodes_num: int = self.graph_nodes_num
        graphs = torch_Tensor(bsz, graph_nodes_num, graph_nodes_num, device=device)
        features = self.features
        len_features: int = len(features)
        mask_edge_types = self.mask_edge_types
        while True:
            if self.example_ptr >= len_features:
                break
            start_id = self.example_ptr
            cur_bsz: int = min(bsz, len_features - start_id)
            cur_batch = features[start_id: start_id + cur_bsz]
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
                context_idxs[i].copy_(torch_Tensor(case.doc_input_ids))
                context_mask[i].copy_(torch_Tensor(case.doc_input_mask))
                segment_idxs[i].copy_(torch_Tensor(case.doc_segment_ids))

                sent_spans = case.sent_spans
                if len(sent_spans) > 0:
                    for j in range(sent_spans[0][0] - 1):
                        query_mapping[i, j] = 1

                sup_para_ids = case.sup_para_ids
                for j, para_span in enumerate(case.para_spans[:para_limit]):
                    is_gold_flag = j in sup_para_ids
                    start, end, _ = para_span
                    if start <= end:
                        end = min(end, max_seq_length-1)
                        is_gold_para[i, j] = int(is_gold_flag)
                        para_mapping[i, start:end+1, j] = 1
                        para_start_mapping[i, j, start] = 1
                        para_end_mapping[i, j, end] = 1

                sup_fact_ids = case.sup_fact_ids
                for j, sent_span in enumerate(sent_spans[:sent_limit]):
                    is_sp_flag = j in sup_fact_ids
                    start, end = sent_span
                    if start <= end:
                        end = min(end, max_seq_length-1)
                        is_support[i, j] = int(is_sp_flag)
                        sent_mapping[i, start:end+1, j] = 1
                        sent_start_mapping[i, j, start] = 1
                        sent_end_mapping[i, j, end] = 1

                answer_candidates_ids = case.answer_candidates_ids
                for j, ent_span in enumerate(case.entity_spans[:ent_limit]):
                    start, end = ent_span
                    if start <= end:
                        end = min(end, max_seq_length-1)
                        ent_mapping[i, start:end+1, j] = 1
                        ent_start_mapping[i, j, start] = 1
                        ent_end_mapping[i, j, end] = 1
                    ans_cand_mask[i, j] = int(j in answer_candidates_ids)

                answer_in_entity_ids = case.answer_in_entity_ids
                is_gold_ent_i = answer_in_entity_ids[0] if len(answer_in_entity_ids) > 0 else IGNORE_INDEX
                is_gold_ent[i] = is_gold_ent_i
                ans_type = case.ans_type
                if ans_type == 0 or ans_type == 3:
                    end_position = case.end_position
                    end_position_0 = end_position[0]
                    if len(end_position) == 0:
                        y1[i] = y2[i] = 0
                    elif end_position_0 < max_seq_length and context_mask[i][end_position_0+1] == 1: # "[SEP]" is the last token
                        y1[i] = case.start_position[0]
                        y2[i] = end_position_0
                    else:
                        y1[i] = y2[i] = 0
                    q_type[i] = ans_type if is_gold_ent_i > 0 else 0
                elif ans_type == 1:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 1
                elif ans_type == 2:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 2
                # ignore entity loss if there is no entity
                if ans_type != 3:
                    is_gold_ent[i].fill_(IGNORE_INDEX)

                qas_id = case.qas_id
                tmp_graph = graph_dict[qas_id]
                for k in range(graph_adj.size(0)):
                    graph_adj[k, k] = 8
                for edge_type in mask_edge_types:
                    graph_adj = torch_where(graph_adj == edge_type, torch_zeros_like(graph_adj), graph_adj)
                graphs[i] = graph_adj

                ids.append(qas_id)

            input_lengths = (context_mask[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            para_mask = (para_mapping > 0).any(1).float()
            sent_mask = (sent_mapping > 0).any(1).float()
            ent_mask = (ent_mapping > 0).any(1).float()

            self.example_ptr += cur_bsz

            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous().to(device),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous().to(device),
                'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous().to(device),
                'context_lens': input_lengths.contiguous().to(device),
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

        self.data_dir = os_path_join(DATASET_FOLDER, 'data_feat')

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
        return os_path_join(self.data_dir, tag, cached_filename)

    def get_example_file(self, tag):
        cached_filename = get_cached_filename('examples', self.config)

        return os_path_join(self.data_dir, tag, cached_filename)

    def get_graph_file(self, tag):
        cached_filename = get_cached_filename('graphs', self.config)
        return os_path_join(self.data_dir, tag, cached_filename)

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
            return gzip_open(file_name, 'rb')
        else:
            return open(file_name, 'rb')

    def __get_or_load__(self, name, file):
        if getattr(self, name) is None:
            with self.get_pickle_file(file) as fin:
                print('loading', file)
                setattr(self, name, pickle_load(fin))

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
        config = self.config
        return self.DataIterator(*self.load_dev(),
                                 bsz=config.eval_batch_size,
                                 device=config.device,
                                 para_limit=config.max_para_num,
                                 sent_limit=config.max_sent_num,
                                 ent_limit=config.max_entity_num,
                                 ans_ent_limit=config.max_ans_ent_num,
                                 mask_edge_types=config.mask_edge_types,
                                 sequential=True)

    @property
    def train_loader(self):
        config = self.config
        return self.DataIterator(*self.load_train(),
                                 bsz=config.batch_size,
                                 device=config.device,
                                 para_limit=config.max_para_num,
                                 sent_limit=config.max_sent_num,
                                 ent_limit=config.max_entity_num,
                                 ans_ent_limit=config.max_ans_ent_num,
                                 mask_edge_types=config.mask_edge_types,
                                 sequential=False)
