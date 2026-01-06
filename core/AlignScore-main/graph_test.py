import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from alignscore import AlignScore
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from alignscore.Parameter import TASK, MODEL_SIZE, LLM_NAME
import torch

test_data_file = 'entities/align_test_max' + str(TASK) + '_data_' + LLM_NAME + '.json'
output_file = 'final_result/' + str(TASK) + '_test_max_'+ MODEL_SIZE + '_' + LLM_NAME + '_Q7toL7.json'
# ckpt_path = 'AlignScore-main/ckpt/' + str(TASK) + '_'+ MODEL_SIZE + '_final.ckpt'
ckpt_path = 'final_models/2_max_dolly_qwen_7b_base_final.ckpt'
mode = 'nli' if TASK == 3 else 'bin'

target_ratios = [-1, 0.0, 0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9, 1.0, 1.1]

with open(test_data_file, 'r') as f:
    data = json.load(f)

scorer = AlignScore(model='model/roberta-' + MODEL_SIZE, 
                    batch_size=32, 
                    device='cuda:0', 
                    ckpt_path=ckpt_path, 
                    evaluation_mode=mode)
 
example_context = {}
example_label = {}
example_current = {}
for item in data:
    context = item['text_a']
    current = item['text_b']
    label = item['label']
    e_id = item['id']
    if e_id in example_context:
        example_context[e_id].append(context)
        example_current[e_id].append(current)
        example_label[e_id].append(label)
    else:
        example_context[e_id] = [context]
        example_current[e_id] = [current]
        example_label[e_id] = [label]

all_labels = []
all_predictions = []

row_labels = []
row_pre = []

graph_labels = []
graph_predictions = [[] for i in target_ratios]
for e_id in example_context:
    context = example_context[e_id]
    label = example_label[e_id]
    current = example_current[e_id]
    score = scorer.score(contexts=context, claims=current)
    if mode == 'bin':
        predictions = [int(align_prob>0.5) for align_prob in score]
    elif mode == 'nli':
        predictions = [align_prob.index(max(align_prob)) for align_prob in score]
        predictions = torch.argmax(torch.tensor(score), dim=1).tolist()

    all_labels.extend([int(item == 1) for item in label])
    row_labels.extend(label)
    all_predictions.extend([int(item == 1) for item in predictions])
    row_pre.extend(predictions)

    if mode== 'bin':
        tmp = 1
        for item in label:
            if item == 1: # 存在一个有幻觉
                tmp = 0 # 整体样本标签为0，表示不正确
                break
        graph_labels.append(tmp)
    elif mode == 'nli':
        tmp = 0
        for item in label:
            if item  == 1:
                tmp = 1
                break
        graph_labels.append(tmp)
    
    if mode == 'bin':
        ratio = predictions.count(1) / len(predictions)
    elif mode == 'nli':
        ratio = predictions.count(1) / len(predictions)

    for index, target_ratio in enumerate(target_ratios):
        if ratio > target_ratio:
            graph_predictions[index].append(0 if mode == 'bin' else 1)
            # graph_predictions[index].append(1)
        else:
            graph_predictions[index].append(1 if mode == 'bin' else 0)
            # graph_predictions[index].append(0)

mean_less = [0,0]
align_less = [0,0]
sent_iner = [0,0]
for label, pre in zip(row_labels, row_pre):
    if (label == 1 and pre == 2):
        mean_less[0] += 1
    elif (label == 2 and pre == 1):
        mean_less[1] += 1
    elif (label == 1 and pre == 0):
        align_less[0] += 1
    elif (label == 0 and pre == 1):
        align_less[1] += 1
    elif (label == 0 and pre == 2):
        sent_iner[0] += 1
    elif (label == 2 and pre == 0):
        sent_iner[1] += 1
print(mean_less, align_less, sent_iner)
result = {}
print('#################### ALL ####################')
f1 = f1_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions)
accuracy = accuracy_score(all_labels, all_predictions)
print({'f1': f1, 'recall': recall, 'precision': precision, 'accuracy': accuracy})
result['all'] = {'f1': f1, 'recall': recall, 'precision': precision, 'accuracy': accuracy}

for index, graph_prediction in enumerate(graph_predictions):
    print('#################### GRAPH {index} ####################'.format(index = target_ratios[index]))
    f1 = f1_score(graph_labels, graph_prediction)
    recall = recall_score(graph_labels, graph_prediction)
    precision = precision_score(graph_labels, graph_prediction)
    accuracy = accuracy_score(graph_labels, graph_prediction)
    print({'f1': f1, 'recall': recall, 'precision': precision, 'accuracy': accuracy})
    result[str(target_ratios[index])] = {'f1': f1, 'recall': recall, 'precision': precision, 'accuracy': accuracy}

with open(output_file, "w") as file:
    json.dump(result, file, indent=4)