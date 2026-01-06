# from core.llm import get_answer
# from retrying import retry
import json
import os
import re
from typing import List
from tqdm import tqdm
import spacy
import stanza

file_dir = 'lrp_result_llama_7b'
hall_path = 'data/response_llama_7b_hallucination.jsonl'
nohall_path = 'data/response_llama_7b_nonhallucination.jsonl'
output_file = 'entities/entities_result_llama_7b.json'

dolly_path = 'data/databricks-dolly-15k_labeled_3b.jsonl'
dolly_output_file = 'entities/entities_result_dolly_qwen_3b.json'

def get_phrases(tree, label):
    if tree.is_leaf():
        return []
    results = [] 
    for child in tree.children:
        results += get_phrases(child, label)
    
    if tree.label == label:
        return [' '.join(tree.leaf_labels())] + results
    else:
        return results

class EntitySelector:
    def __init__(self):
        
        self.nlp = spacy.load('en_core_web_lg')
        self.stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', download_method=None)

    def select_entity(self, sample: str):
        '''
        This function delete time-related information and store it in `time_removed_tweet`.
        '''
        doc = self.nlp(sample)
        stanza_doc = self.stanza_nlp(sample)

        # 名词
        noun_spacy = [ent.text for sent in doc.sents for ent in sent.noun_chunks]
        # 命名实体
        entity_spacy = [ent.text  for sent in doc.sents for ent in sent.ents] 
        # 名词短语
        np_stan = [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'NP')]
        # 动词短语
        vp_stan = [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'VP')]
        # 动词，形容词，副词，名词
        verb_stan = [word.text for sent in stanza_doc.sentences for word in sent.words if word.upos == 'VERB']
        adv_stan = [word.text for sent in stanza_doc.sentences for word in sent.words if word.upos == 'ADV']
        adj_stan = [word.text for sent in stanza_doc.sentences for word in sent.words if word.upos == 'ADJ']
        noun_stan = [word.text for sent in stanza_doc.sentences for word in sent.words if word.upos == 'NOUN']
        
        ents = noun_spacy + entity_spacy + np_stan + vp_stan + verb_stan + adv_stan + adj_stan + noun_stan

        # negation
        negations = [word for word in ['not','never'] if word in sample]

        # look for middle part: relation/ verb
        middle = []
        start_match = ''
        end_match = ''
        for ent in ents:
            # look for longest match string
            if sample.startswith(ent) and len(ent) > len(start_match):
                start_match = ent
            if sample.endswith(ent+'.') and len(ent) > len(end_match):
                end_match = ent
        
        
        if len(start_match) > 0 and len(end_match) > 0:
            
            middle.append(sample[len(start_match):-len(end_match)-1].strip())
            
        # entity = list(set(ents + negations + middle))

        return {
            'noun_spacy': noun_stan,
            'entity_spacy': entity_spacy,
            'np_stan': np_stan,
            'vp_stan': vp_stan,
            'verb_stan': verb_stan,
            'adv_stan': adv_stan,
            'adj_stan': adj_stan,
            'noun_stan': noun_stan,
            'negations': negations,
            'middle': middle,
        }

selector = EntitySelector() 
def get_entity(output_text):
    entity = selector.select_entity(output_text)
    return entity 

def process_document():
    all_data = {}

    data_dict = {}
    data_file = open(hall_path, 'r', encoding='utf-8')
    for line in data_file.readlines():
        dic = json.loads(line)
        data_dict[dic['source_id']] = dic
    
    data_file = open(nohall_path, 'r', encoding='utf-8')
    for line in data_file.readlines():
        dic = json.loads(line)
        data_dict[dic['source_id']] = dic

    file_names = os.listdir(file_dir)
    for file_name in tqdm(file_names):
        example_id = file_name.split('-')[1]
        output_text = data_dict[example_id]['response']
        try:
            result = get_entity(output_text)
        except:
            result = []
            print('Error occurred while getting entities.')
        all_data[example_id] = {'entities': result, 
                                'output_text': output_text, 
                                'lables': data_dict[example_id]['labels']}

    with open(output_file, "w") as file:
        json.dump(all_data, file, indent=4)
        
def process_document_dolly():
    all_data = {}

    data_file = open(dolly_path, 'r', encoding='utf-8')
    for index, line in enumerate(data_file.readlines()):
        dic = json.loads(line)
        output_text = dic['answer']
        prompt = dic["prompt"]
        try:
            result = get_entity(output_text)
        except:
            result = []
            print('Error occurred while getting entities.')

        all_data[index] = {'entities': result, 
                            'output_text': output_text, 
                            'lables': dic["label"]}

    with open(dolly_output_file, "w") as file:
        json.dump(all_data, file, indent=4)

if __name__ == "__main__":
    # process_document()
    process_document_dolly()
    