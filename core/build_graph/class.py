import json
import numpy as np
import torch
text = """The Palestinian Authority has officially become the 123rd member of the International Criminal Court (ICC), giving the court jurisdiction over alleged crimes in Palestinian territories. This includes East Jerusalem and Gaza Strip, which are occupied by Israel. The signing of Rome Statute by Palestinians in January 2021 had already established ICC's jurisdiction over alleged crimes committed \"since June 13, 2014\" in these areas. Now, the court can open a preliminary investigation or formal investigation into the situation in Palestinian territories, potentially leading to war crimes probes against Israeli individuals. However, this could also lead to counter-charges against Palestinians. The ICC welcomed Palestine's accession, while Israel and the US, who are not ICC members, opposed the move.
"""
print(text[219:229])
json_path = 'data/response_llama_7b_hallucination.jsonl'
file = open(json_path, 'r', encoding='utf-8')
label_dict = {'Evident Baseless Info': [],
              'Evident Conflict': [],
              'Subtle Baseless Info': [],
              'Subtle Conflict': []}
for line in file.readlines():
    dic = json.loads(line)
    label_dict[dic['labels'][0]['label_type']].append(dic)
pass
