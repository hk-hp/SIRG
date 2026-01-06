import json
import numpy as np
import torch
from transformers import AutoTokenizer
from lxt.utils import pdf_heatmap, clean_tokens

id = 'rag-15195-1'
model_id = '/home/ /model/llama-2-7b-chat-hf'   
json_path = 'lrp_result_llama_7b/'+ id + '.json'
def heatmap(input_ids, relevance, output):
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

    relevance = relevance / relevance.abs().max()

    # Remove special characters that are not compatible wiht LaTeX
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    tokens = clean_tokens(tokens)

    # Save heatmap as PDF
    pdf_heatmap(tokens, relevance, path=output, backend='xelatex')

with open(json_path, 'r') as f:
    data = json.load(f)
    prompt_ids, response_ids = data["prompt_ids"][0], data["response_ids"][0]
    input_ids = prompt_ids + response_ids
    prompt_length, response_length = len(prompt_ids), len(response_ids)
    relevance = data["relevance"]
    source2dst_relevance = [x[:prompt_length] for x in relevance]
    source_relevance = list(np.array(source2dst_relevance).mean(axis=0))
    source_relevance_max = list(np.array(source2dst_relevance).max(axis=0))
    dst_relevance = list(np.array(source2dst_relevance).mean(axis=-1))
    label = data["hallucination"]
    # torch_relevance = torch.tensor(source_relevance)
    relevance = torch.cat((torch.tensor(source_relevance), torch.zeros(response_length)))

    heatmap(input_ids, relevance, 'heatmap/' + id + '/ARes_mean.pdf')
    relevance = torch.cat((torch.tensor(source_relevance_max), torch.zeros(response_length)))
    heatmap(input_ids, relevance, 'heatmap/' + id + '/ARes_max.pdf')

    for i in range(len(response_ids)):
        input = prompt_ids + response_ids[:i+1]
        intput_relevance = data["relevance"][i][:len(input)]
        intput_relevance[-1] = 0
        heatmap(input, torch.tensor(intput_relevance), 'heatmap/' + id + '/res_' + str(i) + '.pdf')
