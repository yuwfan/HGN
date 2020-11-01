import os
import json
import pandas
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

input_data = json.load(open(input_file, 'r'))
labels, titles, contexts, questions = [], [], [], []

for data in input_data:
    gold_paras = [para for para , _  in data['supporting_facts']]
    for entity, sentences in data['context']:
        label = int(entity in gold_paras)
        title = entity
        context = " ".join(sentences)
        question = data['question']

        labels.append(label)
        titles.append(title)
        contexts.append(context)
        questions.append(question)

df = pandas.DataFrame({'title': titles, 'context': contexts, 'label': labels, 'question':questions})
df.to_csv(output_file)
