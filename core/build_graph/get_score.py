import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
import json
import numpy as np
from transformers import AutoTokenizer, LlamaTokenizerFast
from transformers.models.qwen2 import Qwen2Tokenizer
import numpy as np
import json
import tqdm
import re
import spacy
import string

# 设置数据集路径
dataset_paths = []

file_path = 'entities/entities_result_llama_7b.json'
output_file = 'entities/sent_rel_inf_max_result_llama_7b.json'
lrp_dir = 'lrp_result_llama_7b/'

# dolly_row_path = 'data/databricks-dolly-15k_test.jsonl'
dolly_row_path = 'data/databricks-dolly-15k_labeled_3b.jsonl'
dolly_file_path = 'entities/entities_result_dolly_qwen_3b.json'
dolly_output_file = 'entities/sent_rel_inf_max_result_dolly_qwen_3b.json'

model = 'qwen'
model_id = '/home/ /model/Qwen2.5-3B'

def find_sublist(lst, target):
    pos = []
    for item in range(len(lst) - len(target) + 1):
        if lst[item:item + len(target)] == target:
            pos.append((item, item + len(target)))
    return pos

def get_hallucination_span(response: str, labels: list):
    hall_span = []
    for label in labels:
        before_text = response[:label['start']]

        before_length = len(tokenizer.encode_plus(before_text, add_special_tokens=True).input_ids)
        text_length = len(tokenizer.encode_plus(label['text'], add_special_tokens=False).input_ids)

        hall_span.append((before_length, before_length + text_length, label['label_type']))
    return hall_span

def get_entity_relevance(entity: str, response_ids: list, relevances: list, prompt_length: int, hall_span: list):
    entity_ids = tokenizer.encode_plus(entity, add_special_tokens=False).input_ids
    start_idx = find_sublist(response_ids, entity_ids)
    relevances_list = []
    for item in start_idx:
        start, end = item
        is_hall = False
        if len(hall_span) != 0:
            for span in hall_span:
                if start >= span[0] and end <= span[1]:
                    is_hall = True
                    break
        entity_relevance = np.zeros(len(relevances[0]))
        for i in range(start, end):
            entity_relevance += np.array(relevances[i])
        
        entity_relevance /= (end - start)
        relevances_list.append({
            'entity': entity,
            'is_hallucinated': is_hall,
            'prompt': list(entity_relevance[:prompt_length]), 
            'response': list(entity_relevance[prompt_length:start + prompt_length]),
            'prompt_mean': float(np.mean(entity_relevance[:prompt_length])),
            'response_mean': float(np.mean(entity_relevance[prompt_length:start + prompt_length])),
            })
        # print(f'Entity: {entity}, prompt_mean: {relevances_list[-1]["prompt_mean"]}, response_mean: {relevances_list[-1]["response_mean"]}, is_hallucinated: {is_hall}')
    return relevances_list

# 获取比例分数
def get_score():
    with open(file_path, 'r', encoding='utf-8') as file:
        data: dict = json.load(file)

    example = {}
    for example_id, cotnent in tqdm.tqdm(data.items()):
        is_hall = 0 if len(cotnent['lables']) == 0 else 1
        lrp_file = f'{lrp_dir}rag-{example_id}-{str(is_hall)}.json'
        with open(lrp_file, 'r', encoding='utf-8') as file:
            lrp_data = json.load(file)
        
        entities = cotnent['entities']
        response_ids = lrp_data['response_ids'][0]
        relevances = lrp_data['relevance']
        prompt_length = len(lrp_data['prompt_ids'][0])
        lables = cotnent['lables']
        
        hall_span = get_hallucination_span(lrp_data['response'], lables)
        entity_relevance_list = []
        for entity in entities:
            if len(entity) == 0:
                continue
            entity_relevances = get_entity_relevance(entity, response_ids, relevances, prompt_length, hall_span)
            entity_relevance_list.extend(entity_relevances)
        example[example_id] = {'is_hall': is_hall, 'entity_relevance_list': entity_relevance_list}
    
    with open(output_file, "w") as file: 
         json.dump(example, file, indent=4)
    return example

class BuildInferenceGraph():
    def __init__(self, model):
        self.min_sentence_length = 5
        if model == 'qwen':
            self.nomeaning_ids = [198, 220, 151643] # '\n' ' ' '<end>'
            self.nomeaning_token = ['\n', ' ', '<|endoftext|>']
            self.tokenizer = Qwen2Tokenizer.from_pretrained(model_id, local_files_only=True)
        elif model == 'llama':
            self.nomeaning_ids = [13, 29871, 1] # '\n' '__' '<s>'
            self.tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        self.model = model
        
        self.is_passage = False
        
        self.nlp = spacy.load("en_core_web_lg")

    def get_peace(self, pos, entity, response_off, response_ids, response_tok):
        for index, item in enumerate(response_off):
            if self.model == 'llama':
                if response_tok[index].startswith('▁'):  offset = 1
                else: offset = 0
            elif self.model == 'qwen':
                if response_tok[index].startswith(' '):  offset = 1
                else: offset = 0

            if item[0] + offset == pos:
                break
        if index == len(response_off) - 1: return []

        res = []
        end_pos = response_off[index][0] + len(entity) + offset
        for i in range(index, len(response_off)):
            if end_pos >= response_off[i][1]:
                if response_ids[i] in self.nomeaning_ids or response_tok[i] in string.punctuation:
                    continue
                res.append(i)
            else:
                break
        return res

    def get_sents(self, text: str, ids, offset, token, is_passage=True):
        sents = []
        if is_passage:
            for passage in text.split('\n'):
                if len(passage) != 0:
                    sent_text = passage.strip()
                    if len(sent_text) >= self.min_sentence_length: 
                        pos = text.find(sent_text)
                        sent2ids = self.get_peace(pos, sent_text, offset, ids, token)
                        if len(sent2ids) == 0:
                            sent_text = sent_text.split(' ')[1:]
                            sent_text = ' '.join(sent_text)
                            pos = text.find(sent_text)
                            sent2ids = self.get_peace(pos, sent_text, offset, ids, token)

                        if len(sent2ids) != 0: 
                            sents.append((sent_text, pos, sent2ids, ''))
        else:
            text_offset = 0
            for passage in text.split('\n'):
                if len(passage) != 0:
                    prefix = re.search(r"(passage\s+\d+)", passage, re.DOTALL)
                    doc = self.nlp(passage)
                    for sent in doc.sents:
                        sent_text = sent.text.strip()
                        if len(sent_text) >= self.min_sentence_length: 
                            pos = text.find(sent_text)
                            sent2ids = self.get_peace(pos + text_offset, sent_text, offset, ids, token)
                            if len(sent2ids) == 0:
                                sent_text = sent_text.split(' ')[1:]
                                sent_text = ' '.join(sent_text)
                                pos = text.find(sent_text)
                                sent2ids = self.get_peace(pos + text_offset, sent_text, offset, ids, token)

                            if len(sent2ids) != 0: 
                                prefix_text = ''
                                if prefix is not None and prefix.group(0) not in sent_text:
                                    prefix_text = prefix.group(0) + ': '
                                sents.append((sent_text, pos + text_offset, sent2ids, prefix_text))
                                
                                text_offset += pos + len(sent_text)
                                text = text[pos + len(sent_text):]
        return sents
    
    @staticmethod
    def get_sentence_lable(response_sentence, labels):
        sent_lable = []
        for label in labels:
            if label['start'] <= response_sentence[1] and response_sentence[1] <= label['end']:
                sent_lable.append((label['label_type'], label['meta']))
        return sent_lable
    
    @staticmethod
    def get_sentence_lable_dolly(response_sentence, labels):
        sent_lable = []
        if labels[0] != "CORRECT":
            for label in labels[1]:
                if label in response_sentence[0]:
                    sent_lable.append((labels[0], label))
        return sent_lable
                
    def get_sentence_rel(self, response_sentence, prompts_sentences, word2token, relevances, prompt_length, pre_sentences):    
        sent_start = response_sentence[1]
        sent_end = response_sentence[1] + len(response_sentence[0])

        rel_relevances = []
        for word, word_poses in word2token.items():
            if word not in response_sentence[0]:
                continue

            for word_pos, token_poses in word_poses.items():
                if sent_start <= word_pos and word_pos <= sent_end:
                    for token_pos in token_poses:
                        rel_relevances.append(relevances[token_pos])

        if len(rel_relevances) == 0:
            for item in response_sentence[2]:
                rel_relevances.append(relevances[item])

        rel_relevances = abs(np.array(rel_relevances)).mean(axis=0)
        
        def compute_graph_rel(sent2ids): return np.max(rel_relevances[sent2ids]).item()
        
        inference_head = [rel_relevances[0].item()] # 第一个都是推理头
        
        outer_rel = []
        for prompts_sentence in prompts_sentences:
            sent2ids = prompts_sentence[2]
            outer_rel.append(compute_graph_rel(sent2ids))
            # outer_rel.append(np.mean(rel_relevances[sent2ids]).item())
        
        inner_rel = []
        for pre_sentence in pre_sentences:
            sent2ids = [item + prompt_length for item in pre_sentence]
            inner_rel.append(compute_graph_rel(sent2ids))
            # inner_rel.append(np.mean(rel_relevances[sent2ids]).item())
        
        return inference_head + outer_rel + inner_rel
    
    def compute_rel_llama(self, entities: list, lables: list, relevances: list, prompt: str, response: str):
        prompt_encoder = self.tokenizer.encode_plus(prompt, add_special_tokens=True)
        response_encoder = self.tokenizer.encode_plus(response, add_special_tokens=True)

        prompt_ids = prompt_encoder.input_ids
        response_ids = response_encoder.input_ids
        prompt_off = prompt_encoder._encodings[0].offsets
        response_off = response_encoder._encodings[0].offsets
        prompt_tok = prompt_encoder._encodings[0].tokens
        response_tok = response_encoder._encodings[0].tokens

        word2token = {}
        for entity in entities:
            poses = [substr.start() for substr in re.finditer(re.escape(entity) , response)]
            if len(poses) == 0:
                continue

            pos2peace = {}
            for pos in poses:
                peace = self.get_peace(pos, entity, response_off, response_ids, response_tok)
                if len(peace) != 0: pos2peace[pos] = peace
            word2token[entity] = pos2peace
        
        prompts_sentences = self.get_sents(prompt, prompt_ids, prompt_off, prompt_tok, self.is_passage)
        response_sentences = self.get_sents(response, response_ids, response_off, response_tok, self.is_passage)

        pre_sentences = []
        graph = []
        node_lables = []
        # all_sent_len = len(prompts_sentences) + len(response_sentences)
        for response_sentence in response_sentences:
            sentence_rel = self.get_sentence_rel(response_sentence, prompts_sentences, word2token, relevances, len(prompt_ids), pre_sentences)
            sentence_lable = self.get_sentence_lable(response_sentence, lables)
            pre_sentences.append(response_sentence[2])
            # for i in range(all_sent_len - len(sentence_rel)): sentence_rel.append(0) 
            graph.append(sentence_rel)
            node_lables.append(sentence_lable)
        return graph, node_lables, prompts_sentences, response_sentences

    def compute_rel_qwen(self, entities: list, lables: list, relevances: list, response_tok: list, response_ids: list, prompt: str, response: str):
        # remove <|endoftext|>
        response_tok = response_tok[:-1]  
        relevances = relevances[:-1]  
        response_ids = response_ids[:-1] 
       
        prompt_encoder = self.tokenizer(prompt, add_special_tokens=True)
        prompt_ids = prompt_encoder.input_ids

        prompt_tok = []
        for item in prompt_ids:
            prompt_tok.append(self.tokenizer.decode(item))

        prompt_off = []
        tmp = prompt
        offset = 0
        for item in prompt_tok:
            start = tmp.find(item)
            end = start + len(item)
            prompt_off.append((start + offset, end + offset))
            tmp = tmp[end:]
            offset += end

        response_off = []
        tmp = response
        offset = 0
        response_tok[0] = response_tok[0].strip()
        for item in response_tok:
            start = tmp.find(item)
            end = start + len(item)
            response_off.append((start + offset, end + offset))
            tmp = tmp[end:]
            offset += end

        word2token = {}
        for entity in entities:
            poses = [substr.start() for substr in re.finditer(re.escape(entity) , response)]
            if len(poses) == 0:
                continue

            pos2peace = {}
            for pos in poses:
                peace = self.get_peace(pos, entity, response_off, response_ids, response_tok)
                if len(peace) != 0: pos2peace[pos] = peace
            word2token[entity] = pos2peace
        
        prompts_sentences = self.get_sents(prompt, prompt_ids, prompt_off, prompt_tok, self.is_passage)
        response_sentences = self.get_sents(response, response_ids, response_off, response_tok, self.is_passage)

        pre_sentences = []
        graph = []
        node_lables = []
        for response_sentence in response_sentences:
            sentence_rel = self.get_sentence_rel(response_sentence, prompts_sentences, word2token, relevances, len(prompt_ids), pre_sentences)
            sentence_lable = self.get_sentence_lable_dolly(response_sentence, lables)
            pre_sentences.append(response_sentence[2])
            graph.append(sentence_rel)
            node_lables.append(sentence_lable)
        return graph, node_lables, prompts_sentences, response_sentences

# 获取句子之间的依赖关系拓扑图
def get_sent_rel_llama():
    with open(file_path, 'r', encoding='utf-8') as file:
        data: dict = json.load(file)

    GraphBuilder = BuildInferenceGraph(model='llama')
    example = {}
    index = 0
    for example_id, cotnent in tqdm.tqdm(data.items()):
        index += 1
        is_hall = 0 if len(cotnent['lables']) == 0 else 1
        lrp_file = f'{lrp_dir}rag-{example_id}-{str(is_hall)}.json'
        with open(lrp_file, 'r', encoding='utf-8') as file:
            lrp_data = json.load(file)
        
        entities = cotnent['entities']
        all_entities = list(set(entities['noun_spacy'] +
                                entities['entity_spacy'] + 
                                entities['verb_stan'] + 
                                entities['noun_stan'] + 
                                entities['negations']))
        
        relevances = lrp_data['relevance']
        lables = cotnent['lables']
        
        graph, node_lables, prompts_sentences, response_sentences = GraphBuilder.compute_rel_llama(all_entities, 
                                                                                             lables, relevances, 
                                                                                             lrp_data['prompt'], 
                                                                                             lrp_data['response'])
        example[example_id] = {'is_hall': is_hall, 'graph': graph, 'node_lables': node_lables, 'prompts_sentences': prompts_sentences, 'response_sentences': response_sentences}
    
    with open(output_file, "w") as file: 
         json.dump(example, file, indent=4)
    return example

def get_sent_rel_qwen():
    with open(dolly_file_path, 'r', encoding='utf-8') as file:
        data: dict = json.load(file)

    row_data = []
    data_file = open(dolly_row_path, 'r', encoding='utf-8')
    for line in data_file.readlines():
        dic = json.loads(line)
        row_data.append(dic)

    GraphBuilder = BuildInferenceGraph(model='qwen')
    example = {}

    for example_id, cotnent in tqdm.tqdm(data.items()):
        is_hall = 0 if cotnent['lables'] == 'CORRECT' else 1
        relevances = row_data[int(example_id)]['relevance_scores']
        prompt = row_data[int(example_id)]['prompt']
        response = row_data[int(example_id)]['answer']
        response_tokens = row_data[int(example_id)]['generated_tokens']
        response_ids = row_data[int(example_id)]['generated_token_ids']
        wrong_sents = row_data[int(example_id)]['wrong_sents']
        category = row_data[int(example_id)]['category']
        
        entities = cotnent['entities']
        all_entities = list(set(entities['noun_spacy'] +
                                entities['entity_spacy'] + 
                                entities['verb_stan'] + 
                                entities['noun_stan'] + 
                                entities['negations']))
 
        lables = [cotnent['lables'], wrong_sents]
        if (cotnent['lables'] != 'CORRECT' and len(wrong_sents) == 0) or len(response) == 0:
            print(example_id)
            continue
    
        for i in range(len(relevances)):
            for j in range(len(relevances[-1]) - len(relevances[i])):
                relevances[i].append(0.0)
        
        graph, node_lables, prompts_sentences, response_sentences = GraphBuilder.compute_rel_qwen(all_entities, 
                                                                                             lables, relevances, 
                                                                                             response_tokens, response_ids,
                                                                                             prompt, 
                                                                                             response)
        example[example_id] = {'is_hall': is_hall, 'graph': graph, 'category': category, 'node_lables': node_lables, 'prompts_sentences': prompts_sentences, 'response_sentences': response_sentences}
    
    with open(dolly_output_file, "w") as file: 
         json.dump(example, file, indent=4)
    return example

def get_sent_rel(model):
    if model == 'llama':
        return get_sent_rel_llama()
    elif model == 'qwen':
        return get_sent_rel_qwen()


if __name__ == '__main__':
    get_sent_rel(model=model)