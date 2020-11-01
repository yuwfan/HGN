import spacy
import json
import os
import re
import sys
import itertools

from tqdm import tqdm

input_file = sys.argv[1]
ner_file = sys.argv[2]
output_file = sys.argv[3]

nlp = spacy.load("en_core_web_lg", disable=['parser'])
# ref: https://spacy.io/api/annotation#named-entities
ent_type = set(["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"])
            #"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]

def extract_ner_from_titles(sent, titles, context_ners=None):
    matched = []

    # select candidates for question from all NER in context
    candidates = set()
    if context_ners is not None:
        for doc_ner in context_ners:
            all_ents = itertools.chain.from_iterable(doc_ner[1])
            for ent in all_ents:
                if ent[3] in ent_type:
                    candidates.add(ent[0])

    for title in titles:
        stripped_title = re.sub(r' \(.*?\)$', '', title)
        start_pos = sent.lower().find(stripped_title.lower())
        if start_pos != -1:
            end_pos = start_pos + len(stripped_title)
            # ! use title rather than the matched text in the question
            matched.append((title, start_pos, end_pos, 'TITLE'))

    for word in candidates:
        word = re.sub(r' \(.*?\)$', '', word)
        start_pos = sent.lower().find(word.lower())
        if start_pos != -1:
            end_pos = start_pos + len(word)
            text = sent[start_pos: end_pos]
            matched.append((text, start_pos, end_pos, 'CONTEXT'))

    return matched

def extract_question_ner(full_data):
    print("Extract NER from question")
    all_questions = []
    idx, idx_to_ques = 0, {}

    ques_guid2ner = {}
    for case in full_data:
        guid = case['_id']
        all_questions.append(case['question'])
        idx_to_ques[idx] = guid
        idx += 1

    for idx, doc in enumerate(nlp.pipe(all_questions, n_threads=16, batch_size=1000)):
        guid = idx_to_ques[idx]
        ent_list = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        ques_guid2ner[guid] = ent_list

    return ques_guid2ner

def extract_context_ner(full_data, ner_data=None):
    print("Extract NER from context")
    sentences = []
    sent_cnt = 0
    id_to_sent = {}

    context_guid2ner = {}
    for case in full_data:
        guid = case['_id']
        context_guid2ner[guid] = []
        titles = list(dict(case['context']).keys())

        for title, sents in case['context']:
            context_ner = []
            for sent, sent_ner in zip(sents, ner_data[title]['text_ner']):
                context_ner.append([])
                for ner in sent_ner:
                    if ner[3] in ent_type:
                        context_ner[-1].append(ner)
                # optional
                context_ner[-1].extend(extract_ner_from_titles(sent, titles))

            context_guid2ner[guid].append([title, context_ner])

    return context_guid2ner

data = json.load(open(input_file, 'r'))
# ner_data is from spacy which has been extracted in 0_build_db.py
ner_data = json.load(open(ner_file, 'r'))
ques_guid2ner = extract_question_ner(data)
context_guid2ner = extract_context_ner(data, ner_data)

output_data = {}
for case in data:
    guid = case['_id']
    context = dict(case['context'])
    question_text = case['question']
    titles = context.keys()

    if guid not in output_data:
        output_data[guid] = {}

    # 1. extract context NER from: 1) spacy; 2) title 
    context_ners = context_guid2ner[guid]

    # 2. extract question NER from: 1) spacy; 2) title & ner in context
    ques_ent_1 = ques_guid2ner[guid]
    ques_ent_2 = extract_ner_from_titles(question_text, titles, context_ners)

    output_data[guid]['question'] = ques_ent_1 + ques_ent_2
    output_data[guid]['context'] = context_ners

json.dump(output_data, open(output_file, 'w'))
