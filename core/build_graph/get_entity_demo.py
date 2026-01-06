from core.llm import get_answer
from core.prompt import PROMPTS
from retrying import retry
import json
import re
from typing import List
import re
import nltk
# nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, pos_tag, ne_chunk
from typing import  Dict
import spacy
import stanza

sys_prompt = """You are an AI assistant that helps people find important entities in the ginven context and output the entities in a json array ['entity1', 'entity2', ...]."""
# sys_prompt = """You are an AI assistant that helps people find entities with information in the ginven context and output the entity in a json array ['entity1', 'entity2', ...]."""
# sys_prompt = """You are an AI assistant that helps people find Prepositions or Phrases without practical meaning in the ginven context and output the them in a json array ['entity1', 'entity2', ...]."""
output_text = "Sure! Here's how to plant a potato that has sprouted based on the provided passages:\n\nPassage 1 mentions cutting large, sprouted potatoes into chunks with two or three eyes each, using a sharp knife. It recommends placing the potato chunks on a tray or cookie sheet and setting them in a cool, dark location overnight to allow the cut ends to dry out before planting. This step is important as it helps prevent rot and ensures the potatoes have the best chance of growing successfully.\n\nPassage 2 provides an overview of the process of planting seed potatoes, including the fact that they go through three stages: young seed, middle aged, and old. However, it does not provide specific instructions for planting sprouted potatoes.\n\nPassage 3 provides more detailed instructions for planting potatoes, including turning the soil over in the planned planting area, digging a trench, and cutting the potatoes into cubes with a sprout on each cube. It recommends covering the potatoes with a layer of compost and replacing the dirt on top of the plantings.\n\nBased on these passages, here are the steps to plant a sprouted potato:\n\n1. Cut the sprouted potato into chunks with two or three eyes each, using a sharp knife.\n2. Allow the cut ends to dry out in a cool, dark location overnight.\n3. Dig a trench about 4 inches wide and 6 to 8 inches deep in the planned planting area.\n4. Place the dried-out potato chunks into the bottom of the trench, separating each chunk by about 6 to 8 inches.\n5. Cover the potatoes with a layer of compost.\n6. Replace the dirt on top of the plantings.\n\nI hope this helps! Let me know if you have any further questions."

@retry(stop_max_attempt_number=3)
def llm_get_entity(output_text):
    examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: 1]
        )
    prompt = PROMPTS["entity_extraction"].format(examples=examples, input_text=output_text)
    output = get_answer(prompt)
    print(output)
    result: List[str] = re.search(r"\[.*\]", output, re.DOTALL).group(0)[1: -1].split(',')
    final_res = []
    for i in range(len(result)):
        world = result[i].strip()[1:-1]
        if world in output_text:
            final_res.append(world)
    return final_res

def nltk_get_entity(text):
    
    # 分词和词性标注
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    # 命名实体识别
    named_entities = ne_chunk(pos_tags)
    word_type = {}
    for item in named_entities:
        if item[1] in word_type:
            word_type[item[1]].append(item[0])
        else:
            word_type[item[1]] = [item[0]]
    # 输出结果
    print(named_entities)

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


if __name__ == "__main__":
    """try:
        result = get_entity(output_text)
    except:
        result = []
    print(result)"""
    # nltk_get_entity(output_text)
    selector = EntitySelector() 
    # output_text = 'Night of the Living Dead is a Spanish comic book.'
    entity = selector.select_entity(output_text)
    print(entity)
    pass