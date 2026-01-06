import torch
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm

file_path = 'entities/score_result_llama_7b.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data: dict = json.load(file)

for example_id, example in tqdm(data.items()):
    pos_x = []
    pos_y = []
    neg_x = []
    neg_y = []

    is_hallucinated_example = example['is_hall']
    entity_list = example['entity_relevance_list']

    for item in entity_list:
        is_hallucinated = item['is_hallucinated']

        # prompt_mean = item['prompt_mean']
        # response_mean = item['response_mean']
        prompt_mean = max(item['prompt'])
        response_mean = max(item['response'])

        # prompt_rato = prompt_mean / (prompt_mean + response_mean + 1e-10)
        # response_rato = response_mean / (prompt_mean + response_mean + 1e-10)
        prompt_rato = prompt_mean
        response_rato = response_mean
        if is_hallucinated:
            pos_x.append(prompt_rato)
            pos_y.append(response_rato)
        else:
            neg_x.append(prompt_rato)
            neg_y.append(response_rato)

    plt.scatter(pos_x, pos_y, c='red', label='pos', s=10, alpha=0.2)
    plt.scatter(neg_x, neg_y, c='blue', label='neg', s=10, alpha=0.2)
    plt.savefig(f'scatter/{is_hallucinated_example}_{example_id}_mean.png')
    plt.clf()
pass