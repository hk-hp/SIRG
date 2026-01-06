import re
import nltk
# nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, pos_tag, ne_chunk

def extract_entities(text):
    
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

text = "Sure! Here's how to plant a potato that has sprouted based on the provided passages:\n\nPassage 1 mentions cutting large, sprouted potatoes into chunks with two or three eyes each, using a sharp knife. It recommends placing the potato chunks on a tray or cookie sheet and setting them in a cool, dark location overnight to allow the cut ends to dry out before planting. This step is important as it helps prevent rot and ensures the potatoes have the best chance of growing successfully.\n\nPassage 2 provides an overview of the process of planting seed potatoes, including the fact that they go through three stages: young seed, middle aged, and old. However, it does not provide specific instructions for planting sprouted potatoes.\n\nPassage 3 provides more detailed instructions for planting potatoes, including turning the soil over in the planned planting area, digging a trench, and cutting the potatoes into cubes with a sprout on each cube. It recommends covering the potatoes with a layer of compost and replacing the dirt on top of the plantings.\n\nBased on these passages, here are the steps to plant a sprouted potato:\n\n1. Cut the sprouted potato into chunks with two or three eyes each, using a sharp knife.\n2. Allow the cut ends to dry out in a cool, dark location overnight.\n3. Dig a trench about 4 inches wide and 6 to 8 inches deep in the planned planting area.\n4. Place the dried-out potato chunks into the bottom of the trench, separating each chunk by about 6 to 8 inches.\n5. Cover the potatoes with a layer of compost.\n6. Replace the dirt on top of the plantings.\n\nI hope this helps! Let me know if you have any further questions."
extract_entities(text)