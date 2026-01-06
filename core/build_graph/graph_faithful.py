import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from transformers.models.llama import LlamaTokenizer, modeling_llama
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


model_id = '/home/ /model/llama-2-7b-chat-hf'
graph_path = 'entities/sent_rel_inf_max_result_llama_7b.json'

model = modeling_llama.LlamaForCausalLM.from_pretrained(
    model_id,
    device_map='cuda',
    torch_dtype=torch.bfloat16
)
device = torch.device("cuda")
model.eval()
tokenizer = LlamaTokenizer.from_pretrained(model_id)

def lrp(all_context, input_text, output_text, temperature=1, max_steps=5000):
    all_ids = tokenizer.encode_plus(all_context, return_tensors="pt", add_special_tokens=True).input_ids
    all_ids = all_ids.to(device)
    input_ids = tokenizer.encode_plus(input_text, return_tensors="pt", add_special_tokens=True).input_ids
    input_ids=input_ids.to(device)
    output_ids = tokenizer.encode_plus(output_text, return_tensors="pt", add_special_tokens=False).input_ids

    output_size = output_ids.shape[-1]
    logit_errs = []
    prob_errs = []
    for gen_length in range(min(output_size, max_steps)):
        output_token_id = output_ids[:, gen_length].to(device)
        
        output_logits = model(all_ids)['logits'][0, -1, :]
        y0 = output_logits.detach().cpu().float().numpy()
        probs = torch.softmax(output_logits / temperature, dim=-1)
        # rank = torch.where(probs.argsort(descending=True) == output_token_id)[0].item()

        changed_output_logits = model(input_ids)['logits'][0, -1, :]
        y = changed_output_logits.detach().cpu().float().numpy()
        changed_probs = torch.softmax(changed_output_logits / temperature, dim=-1)
        # changed_rank = torch.where(changed_probs.argsort(descending=True) == output_token_id)[0].item()

        logit_err = np.sum((y0[output_token_id] - y[output_token_id])**2)
        prob_err = abs(probs[output_token_id] - changed_probs[output_token_id]).detach().cpu().float().numpy()[0]

        logit_errs.append(logit_err)
        prob_errs.append(prob_err)

        # update input
        output_token_id = output_ids[:, gen_length].to(device)
        nxt_token_id = torch.tensor([[output_token_id]]).to(device)
        input_ids = torch.cat((input_ids, nxt_token_id), dim=-1)
        all_ids = torch.cat((all_ids, nxt_token_id), dim=-1)

        torch.cuda.empty_cache()

    sent_logit_err = sum(logit_errs) / len(logit_errs)
    sent_prob_err = sum(prob_errs) / len(prob_errs)
    return sent_logit_err, sent_prob_err

def rel_lrp(graph, prompts_sentences, response_sentences, node_labels):
    pass

def sent_lrp(graph, prompts_sentences, response_sentences, node_labels, max_sentence_num, mode='prun'):
    all_logit = []
    all_prob = []
    for index, response in tqdm(enumerate(response_sentences), total=len(response_sentences)):
        edges = graph[index]
        sents = prompts_sentences + response_sentences[:index]
        edges_sort = np.argsort(-np.array(edges)) # 从大到小

        all_context = ""
        for item in sents: all_context += item[0] + ' '
        all_context = all_context[:-1]

        sent_logit_errs = [] 
        sent_prob_errs = []
        for i in range(1, max_sentence_num + 1):
            context = ""
            for j in range(len(sents)):
                if j not in edges_sort[:i]:
                    context += sents[j][0] + ' '
            sent_logit_err, sent_prob_err = lrp(all_context, context[:-1], response[0])
            sent_logit_errs.append(sent_logit_err)
            sent_prob_errs.append(sent_prob_err)
        all_logit.append(sent_logit_errs)
        all_prob.append(sent_prob_errs)
    return all_logit, all_prob

if __name__ == '__main__':
    max_sentence_num = 10
    mode = 'prun'
    with open(graph_path, 'r') as f: 
        graphs: dict = json.load(f)
    for g_id, item in graphs.items():
        graph = item['graph']
        prompts_sentences = item['prompts_sentences']
        response_sentences = item['response_sentences']
        node_labels = item['node_lables']
        if len(prompts_sentences) < 10:
            continue
        all_logit, all_prob = sent_lrp(graph, 
                                       prompts_sentences, 
                                       response_sentences, 
                                       node_labels, 
                                       max_sentence_num, 
                                       mode=mode)
        all_logit = np.array(all_logit)
        all_prob = np.array(all_prob)
        mean_logit = np.mean(all_logit, axis=0)
        mean_prob = np.mean(all_prob, axis=0)

        f, axs = plt.subplots(1, 2, figsize=(14, 8))
        axs[0].plot(mean_logit)
        axs[0].set_title('($y_0$-$y_p$)$^2$')
        axs[1].plot(mean_prob)
        axs[1].set_title('logits$_k$')    
        plt.legend()
        plt.savefig('2.png')
        exit()
    pass

