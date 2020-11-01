import spacy
import json
import os
import re
import torch
import numpy as np
import sys

from tqdm import tqdm
from collections import Counter

assert len(sys.argv) == 6

raw_data = json.load(open(sys.argv[1], 'r'))
doc_link_data = json.load(open(sys.argv[2], 'r'))
ent_data = json.load(open(sys.argv[3], 'r'))
para_data = json.load(open(sys.argv[4], 'r'))
output_file = sys.argv[5]

def select_titles(question_text, question_entities):
    def custom_key(x):
        # x[1]: start position
        # x[2]: end position
        return x[1], -x[2]

    # get TITLE entities
    candidates = []
    title_set = set()
    for ent in question_entities:
        # only keep the entities from title matching
        if ent[3] != 'TITLE' :
            continue

        stripped_ent = re.sub(r' \(.*?\)$', '', ent[0])
        if stripped_ent in title_set:
            continue

        title_set.add(stripped_ent)
        candidates.append(ent)

    # If match multiple titles with the same start, then take the longest one
    sorted_candidiates = sorted(candidates, key=custom_key)

    non_overlap_titles, overlapped_titles = [], []
    question_mask = [0] * len(question_text)

    for i in range(len(sorted_candidiates)):
        start_pos, end_pos = sorted_candidiates[i][1], sorted_candidiates[i][2]
        is_masked = False
        for p in range(start_pos, end_pos):
            if question_mask[p] == 1:
                is_masked = True
        if not is_masked:
            non_overlap_titles.append(sorted_candidiates[i][0])
        else:
            overlapped_titles.append(sorted_candidiates[i][0])
        for p in range(start_pos, end_pos):
            question_mask[p] = 1

    return non_overlap_titles, overlapped_titles

def build_dict(title_list):
    title_to_id, id_to_title = {}, {}

    for idx, title in enumerate(title_list):
        id_to_title[idx] = title
        title_to_id[title] = idx

    return title_to_id, id_to_title

def build_title_to_entities(context, filter_ent_type=[]):
    title_to_ent = {}

    for title, sent_ent_list in context:
        title_to_ent[title] = set()
        for sent_ent in sent_ent_list:
            for ent in sent_ent:
                if ent[3] not in filter_ent_type:
                    title_to_ent[title].add(ent[0].lower())
    return title_to_ent

def build_PG(titles):
    # build hyperlink graph
    N = len(titles)
    para_adj = np.zeros((N, N), dtype=np.float32)

    title_to_id, id_to_title = build_dict(titles)

    for title in titles:
        sent_links = doc_link_data[title]['hyperlink_titles']
        for next_title in [next_title for sent_link in sent_links for next_title in sent_link]:
            if next_title in titles:
                pi, pj = title_to_id[title], title_to_id[next_title]
                para_adj[pi, pj] = 1

    return para_adj

def bfs_step(start_vec, graph):
    """
    :param start_vec:   [E]
    :param graph:       [E x E]
    :return: next_vec:  [E]
    """
    next_vec = torch.matmul(start_vec.float().unsqueeze(0), graph)
    next_vec = (next_vec > 0).long().squeeze(0)
    return next_vec

para_num = []
selected_para_dict = {}

for case in tqdm(raw_data):
    guid = case['_id']
    context = dict(case['context'])
    para_scores = para_data[guid]
    selected_para_dict[guid] = []

    if len(para_scores) == 0:
        print(guid)
        continue

    title_to_id, id_to_title = build_dict(context.keys())
    sel_para_idx = [0] * len(context)

    # question entity matching
    question_entities = ent_data[guid]['question']

    # hop 1.1 title in ques and top rank
    sel_titles, _ = select_titles(case['question'], question_entities)
    for idx, (para, score) in enumerate(para_scores):
        if para in sel_titles and idx < 2:
            sel_para_idx[title_to_id[para]] = 1

    # hop 1.2:  if cannot match by title, match entities
    title_to_ent = build_title_to_entities(ent_data[guid]['context'], filter_ent_type=['CONTEXT'])
    if sum(sel_para_idx) == 0:
        linked_title = None
        for idx, (title, score) in enumerate(para_scores):
            if title not in title_to_id:
                continue
            ent_set = title_to_ent[title] # all entities from this document
            for ent in question_entities:
                if ent[0].lower() in ent_set:
                    linked_title = title
            if linked_title is not None: # stop finding if match with question entities
                break
        if linked_title is None: # use default one
            assert len(para_scores)  > 0
            linked_title = para_scores[0][0]
        sel_para_idx[title_to_id[linked_title]] = 1

    selected_para_dict[guid].append([id_to_title[i] for i in range(len(context)) if sel_para_idx[i] == 1])

    # second hop: use hyperlink
    second_hop_titles = []
    para_adj = build_PG(context.keys())

    if sum(sel_para_idx) == 1:
        next_titles = []
        next_vec = bfs_step(torch.tensor(sel_para_idx), torch.from_numpy(para_adj))
        next_vec_list = next_vec.nonzero().squeeze(1).numpy().tolist()
        for sent_id in next_vec_list:
            next_titles.append(id_to_title[sent_id])

        # hop 2.1: select the highest score for next title
        # 1. define the next and default title for second hop
        # 2. enumerate all docs, if found next link then stop
        linked_title, default_title = None, None 
        for para, score in para_scores:
            if linked_title is not None: 
                break
            if para not in title_to_id: # skip documents that are not in the supporting docs
                continue

            if sel_para_idx[title_to_id[para]] == 0: # only deal with the ones have not been selected
                if default_title is None:
                    default_title = para

                if para in next_titles:
                    linked_title = para
        linked_title = default_title if linked_title is None else linked_title
        if linked_title is not None:
            sel_para_idx[title_to_id[linked_title]] = 1
            second_hop_titles = [linked_title]

    selected_para_dict[guid].append(second_hop_titles)

    # others, keep a high recall
    other_titles = []
    for para, score in para_scores:
        if para not in title_to_id:
            continue
        if sum(sel_para_idx) == 4:
            break
        ind = title_to_id[para]
        if sel_para_idx[ind] == 0:
            sel_para_idx[ind] = 1
            other_titles.append(para)
    selected_para_dict[guid].append(other_titles)
    para_num.append(sum(sel_para_idx))

json.dump(selected_para_dict, open(output_file, 'w'))
