from sys import argv
from re import sub as re_sub
from string import punctuation
from collections import Counter
from ujson import load as ujson_load


EXCLUDE = set(punctuation)
ZERO_METRIC = (0.0, 0.0, 0.0)

def normalize_answer(s):

    def remove_articles(text):
        return re_sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in EXCLUDE)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    is_unequal: bool = normalized_prediction != normalized_ground_truth
    if normalized_prediction in ['yes', 'no', 'noanswer'] and is_unequal:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and is_unequal:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec_recall: float = prec + recall
    f1: float = 2 * prec * recall / prec_recall if prec_recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall

def eval(prediction_file, gold_file):
    with open(prediction_file) as f:
        prediction = ujson_load(f)
    with open(gold_file) as f:
        gold = ujson_load(f)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}

    answer = prediction['answer']
    sp = prediction['sp']
    total = 0
    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        if cur_id not in answer:
            #print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, answer[cur_id], dp['answer'])
        if cur_id not in sp:
            #print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, sp[cur_id], dp['supporting_facts'])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            prec_recall = joint_prec + joint_recall
            if prec_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / prec_recall
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

            total += 1

    for k in metrics.keys():
        metrics[k] /= total

    return metrics

if __name__ == '__main__':
    print(eval(argv[1], argv[2]))

