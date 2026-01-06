import os.path
import json
import numpy as np
from prompt import PROMPTS
import re

from utils import generate

output_file = 'entities/llm_response.json'
json_path = 'entities/pass_rel_result_llama_7b.json'

def calculate_metrics(TN, TP, FN, FP):
    # 计算准确率
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # 计算精确率
    precision = TP / (TP + FP)
    # 计算召回率
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    }

def process_data(top_k = 3):
    with open(json_path, 'r') as f:
        data = json.load(f)

    process_data = {}
    for example_id, example in data.items():      
        graph =  example['graph']
        node_lables = example['node_lables']
        prompts_sentences = example['prompts_sentences']
        response_sentences = example['response_sentences']

        edge_weights = [] 
        graph_max = [max(node) for node in graph]
        globle_ratio = [node_max / sum(graph_max) for node_max in graph_max]
        node_std = []
        for index, (node, ratio) in enumerate(zip(graph, globle_ratio)):
            node = np.array(node)
            connect = node / sum(node)
            # connect = node / sum(node) * ratio * 100
            edge_weights.append(connect)
            node_std.append(np.std(connect))
       
        all_sentences = prompts_sentences + response_sentences
        data_inform = []
        for index, edge_weight in enumerate(edge_weights):
            edge_indexes = np.argsort(edge_weight)[::-1][:top_k]
            
            source_sentence = []
            for edge_index in edge_indexes:
                if edge_index < len(prompts_sentences):
                    is_prompt = True
                    sentence = all_sentences[edge_index][3] + all_sentences[edge_index][0]
                else:
                    is_prompt = False
                    sentence = "" + all_sentences[edge_index][0]
                source_sentence.append({
                    'sentence': sentence,
                    'is_prompt': is_prompt,
                    'weight': edge_weight[edge_index]
                    })
                
            data_inform.append({
                'source_sentences': source_sentence,
                'target_sentence': response_sentences[index][0],
                'pre_senctence': all_sentences[index + len(prompts_sentences) - 1][0],
                'lable': 0 if len(node_lables[index]) == 0 else 1
            })
        
        process_data[example_id] = data_inform
    return process_data

def compute_sent(sent):
    context = ""
    for i, source in enumerate(sent['source_sentences']):
        context += PROMPTS['context_prompt'].format(num=1+i, 
                                                    context=source['sentence'], 
                                                    contribution=round(source['weight'], 2))
    prompt = PROMPTS["hallucination_detect_prompt_llm"].format(context=context, 
                                                  # previous=sent['pre_senctence'], 
                                                  current=sent['target_sentence'])
    result = generate(prompt) + '\n}'
    try:
        output = json.loads(re.search(r"\{.*\}", result, re.DOTALL).group(0))['Hallucination']
        if type(output) is str:
            if "true" in result.lower(): output = True
            elif "false" in result.lower(): output = False
        return output, result, prompt
    except Exception as e:
        print("Error in generating response:", e)
        if "true" in result.lower(): output = True
        elif "false" in result.lower(): output = False
        else: 
            output = None
            print("None")
        return output, result, prompt

    

def compute_graph(sent_results):
    for sent_result in sent_results:
        if sent_result[0] is not None and sent_result[0] == True:
            return True
    return False

if __name__ == '__main__':
    TN, TP, FN, FP = 0, 0, 0, 0

    data = process_data(top_k=3)

    already_judged = {}
    for example_id, graph in data.items():
        sent_results = []
        for sent in graph:
            sent_results.append(compute_sent(sent))
        
        graph_result = compute_graph(sent_results)

        if "yes" in text.lower():
            record["judge"] = "yes"
            if record["label"] == "CORRECT":
                print("no hallucination, judge wrong")
                FN += 1
            else:
                print("exists hallucination, judge correct")
                TN += 1
        elif "no" in text.lower():
            record["judge"] = "no"
            if record["label"] == "CORRECT":
                print("no hallucination, judge correct")
                TP += 1
            else:
                print("exists hallucination, judge wrong")
                FP += 1

    with open(output_file, "w") as file:
        json.dump(already_judged, file, indent=4)
    print(calculate_metrics(TN, TP, FN, FP))
