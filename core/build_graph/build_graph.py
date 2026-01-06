from __future__ import division
import matplotlib.pyplot as plt
import networkx as nx
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

json_path = 'entities/sent_rel_result_max_result_llama_7b.json'
heatmap_dir = 'graph_heatmap/'

def get_node_text():
    with open(json_path, 'r') as f:
        data = json.load(f)

    node_text = {}
    for example_id, example in data.items():
        prompts_sentences = example['prompts_sentences']
        response_sentences = example['response_sentences']
        node = {}
        for index, sent in enumerate(prompts_sentences):
            node['p' + str(index)] = sent[0]
        for index, sent in enumerate(response_sentences):
            node['r' + str(index) + '_' + str(index + len(prompts_sentences))] = sent[0]

        for index, item in enumerate(example['node_lables']):
            if len(item) > 0:
                node['L' + str(index)] = item
        node_text[example_id] = node
        
    with open('entities/node_llama_7b.json', "w") as file:
        json.dump(node_text, file, indent=4)
    return node_text

def process_data():
    with open(json_path, 'r') as f:
        data = json.load(f)

    num = 0
    for example_id, example in data.items():
        num += 1
        if example_id != '15127':
            continue
            
        graph =  example['graph']
        node_lables = example['node_lables']
        prompts_sentences = example['prompts_sentences']
        response_sentences = example['response_sentences']

        lables = '_'
        for index, lable in enumerate(node_lables):
            if len(lable) != 0:
                lables += str(index) + '_'

        edge_weight = [] 
        edges = []
        graph_max = [max(node) for node in graph]
        globle_ratio = [node_max / sum(graph_max) for node_max in graph_max]
        node_std = []
        for index, (node, ratio) in enumerate(zip(graph, globle_ratio)):
            node = np.array(node)
            connect = node / sum(node)
            # connect = node / sum(node) * ratio * 100
            edge_weight.append(connect)
            node_std.append(np.std(connect))

            for i in range(len(connect)):
                if connect[i] > 1:
                    edges.append((i + 1, index + len(prompts_sentences) + 1, connect[i]))
        heatmap_plot(edge_weight, example_id, lables)
    return data

def heatmap_plot(data, name, lables):
    if len(lables) == 1:
        ishall = '0'
    else:
        ishall = '1'

    if type(data) is list:
        for i in range(len(data)):
            item = data[i]
            pad = data[-1].shape[0] - item.shape[0]
            if pad > 0:
                data[i] = np.concat((item, np.zeros(pad)), axis=0)
        data = np.array(data)
    plt.figure(figsize=(8, 6))
    # sns.heatmap(data, annot=True, fmt=".2f", cmap="coolwarm")
    # sns.heatmap(data, annot=True, fmt=".1f", cmap="coolwarm")
    sns.heatmap(data, cmap="coolwarm")
    plt.title(lables)
    plt.savefig(heatmap_dir + ishall + '_' + name + '_heatmap.png')

def network_plot():
    G = nx.Graph()

    G.add_edge('a', 'b', weight=1)
    G.add_edge('a', 'c', weight=2)
    G.add_edge('c', 'd', weight=3)
    G.add_edge('c', 'e', weight=4)
    G.add_edge('c', 'f', weight=5)
    G.add_edge('a', 'd', weight=6)
    pos = nx.spring_layout(G)

    M = G.number_of_edges()
    labels = nx.get_edge_attributes(G, 'weight')
    alpha=[]
    for u, v, d in G.edges(data=True):
        alpha.append(d['weight'] / max(labels.values()))

    nodes = nx.draw_networkx_nodes(G, pos, node_color='red')
    edges = nx.draw_networkx_edges(G, pos, arrowstyle='->', arrows=True,
                                arrowsize=10, edge_color='blue',
                                alpha=alpha,
                                edge_cmap=plt.cm.Blues, width=2)
    # set alpha value for each edge

    ax = plt.gca()
    ax.set_axis_off()
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)

    plt.savefig('networkx_graph.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # get_node_text()
    # process_data()
    network_plot()