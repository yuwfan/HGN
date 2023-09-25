from sys import argv
from json import load as json_load
from pandas import DataFrame

input_file = argv[1]
output_file = argv[2]

input_data = json_load(open(input_file, 'r'))
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

df = DataFrame({'title': titles, 'context': contexts, 'label': labels, 'question':questions})
df.to_csv(output_file)
