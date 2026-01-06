import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
import torch
import json_lines
import json
from tqdm import tqdm
import torch
from transformers.models.llama import LlamaTokenizer, modeling_llama
from lxt.efficient import monkey_patch
import jsonlines
from tqdm import tqdm
import json


model_id = 'model/llama-2-7b-chat-hf'
graph_path = 'entities/sent_rel_inf_max_result_llama_7b.json'

monkey_patch(modeling_llama, verbose=True)
model = modeling_llama.LlamaForCausalLM.from_pretrained(
    model_id,
    device_map='cuda',
    torch_dtype=torch.bfloat16
)
# Deactivate gradients on parameters

for param in model.parameters():
    param.requires_grad = False

# Optionally enable gradient checkpointing (2x forward pass)
model.train()
model.gradient_checkpointing_enable()

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_id)
eos_token = '</s>'
device = torch.device("cuda")


def append_backward_hooks(m):
    rel_layer = {}

    def generate_hook(layer_name):
        def backward_hook(module, input_grad, output_grad):
            # cloning the relevance makes sure, it is not modified through memory-optimized LXT inplace operations if used
            rel_layer[layer_name] = output_grad[0].clone()

        return backward_hook

    # append hook to last activation of mlp layer
    for name, layer in m.named_modules():
        layer.register_full_backward_hook(generate_hook(name))
    return rel_layer


def read_response(path, label=0):
    responses = []
    with json_lines.open(path) as jsonl_file:
        for i, json_line in enumerate(jsonl_file):
            responses.append({
                "hallucination": label,    
                "response": json_line["response"],
                "source_id": json_line["source_id"],
                "temperature": json_line["temperature"]
            })
    return responses


def read_source(path):
    sources = []
    with json_lines.open(path) as jsonl_file:
        for i, json_line in enumerate(jsonl_file):
            sources.append({
                "source_id": json_line["source_id"],
                "prompt": json_line["prompt"],
                "question": json_line["source_info"]["question"]
            })
    return sources


def check_and_zip(sources: list, responses: list):
    assert len(sources) == len(responses)
    sources.sort(key=lambda x: x["source_id"])
    responses.sort(key=lambda x: x["source_id"])
    result = []
    for (source, response) in zip(sources, responses):
        assert source["source_id"] == response["source_id"]
        result.append({
            "source_id": source["source_id"],
            "prompt": source["prompt"],
            "response": response["response"],
            "hallucination": response["hallucination"],
            "temperature": response["temperature"]
        })
    return result

def dump_json(data, file_name):
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)


def get_lxt_relevance(input_text, output_text, temperature=1, max_steps=500):
    # input_text = """## Context: Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall. As of November 2022, 310 people have died on Everest. Over 200 bodies remain on the mountain and have not been removed due to the dangerous conditions. The first recorded efforts to reach Everest's summit were made by British mountaineers. As Nepal did not allow foreigners to enter the country at the time, the British made several attempts on the north ridge route from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m (22,970 ft) on the North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), marking the first time a human had climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one of the greatest mysteries on Everest to this day: George Mallory and Andrew Irvine made a final summit attempt on 8 June but never returned, sparking debate as to whether they were the first to reach the top. Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, using the southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 Swiss expedition. The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first reported ascent of the peak from the north ridge on 25 May 1960. ## Question: How high did they climb in 1922? ## Instruct: Please think step by step first, then output the process of thinking and answer the Question according to Context. """
    input_ids = tokenizer.encode_plus(input_text, return_tensors="pt", add_special_tokens=True).input_ids
    input_ids_origin=input_ids.clone()
    input_ids=input_ids.to(device)
    output_ids = tokenizer.encode_plus(output_text, return_tensors="pt", add_special_tokens=True).input_ids

    input_size, output_size = input_ids.shape[-1], output_ids.shape[-1]
    relevances = []

    for gen_length in tqdm(range(min(output_size, max_steps))):
        input_embeds = model.get_input_embeddings()(input_ids)
        output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
        # 引入温度参数
        next_token_logits = output_logits[0, -1, :] / temperature
        # 对logits应用softmax得到概率分布
        probs = torch.softmax(next_token_logits, dim=-1)
        # output token id
        llm_token = tokenizer.decode(torch.multinomial(probs, num_samples=1).item())

        # check output token rank
        # output token
        # 获取排序后的索引
        sorted_indices = probs.argsort(descending=True)
        # sorted = probs.sort(descending=True)
        output_token_id = output_ids[:, gen_length].to(device)
        output_token = tokenizer.decode(output_token_id)
        rank = torch.where(sorted_indices == output_token_id)[0].item()
        print(f'{gen_length+1}th token, token={output_token}, rank={rank}, llm={llm_token}', end="\n")

        # update input
        nxt_token_id = torch.tensor([[output_token_id]])
        nxt_token_id = nxt_token_id.to(device)
        input_ids = torch.cat((input_ids, nxt_token_id), dim=-1)
        # input_ids = torch.cat((input_ids, sorted_indices[0].reshape(1,1)), dim=-1)

        # Get LXT relevance
        target_logits = next_token_logits[output_token_id]
        target_logits.backward(target_logits)
        relevance = input_embeds.grad.float().sum(-1).cpu()[0]

        # extend to same length
        relevance=relevance.cpu().tolist()
        relevance += [1] + [0] * (input_size + output_size - len(relevance) - 1)
        assert len(relevance) == input_size + output_size
        relevances.append(relevance)
    return input_ids_origin.cpu().tolist(),output_ids.tolist(),relevances


if __name__ == '__main__':
    qa_source = read_source("data/QA.jsonl")
    qa_hallucinated_response = read_response("data/response_llama_7b_hallucination.jsonl", label=1)
    qa_nonhallucinated_response = read_response("data/response_llama_7b_nonhallucination.jsonl", label=0)
    qa_pairs = check_and_zip(qa_source, qa_hallucinated_response + qa_nonhallucinated_response)
    for qa_pair in qa_pairs:
        input_ids,output_ids,qa_relevance=get_lxt_relevance(qa_pair["prompt"], qa_pair["response"],temperature=qa_pair["temperature"])
        dump_json({
            "prompt_ids":input_ids,
            "response_ids":output_ids,
            "relevance":qa_relevance,
            "source_id":qa_pair["source_id"],
            "prompt":qa_pair["prompt"],
            "response":qa_pair["response"],
            "temperature":qa_pair["temperature"],
            "hallucination":qa_pair["hallucination"]
        },f"./rag-{qa_pair['source_id']}-{qa_pair['hallucination']}.json")

