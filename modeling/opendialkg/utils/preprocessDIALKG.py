import json
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import collections
import torch
from collections import defaultdict
from utils.hugging_face import SPECIAL_TOKENS,MODEL_INPUTS, PADDED_INPUTS, PADDED_SPECIAL, build_input_from_segments, get_loader,test_dataloader
from utils.eval_metrics import get_global_entity_KVR
import csv
import pandas as pd
import numpy as np
import copy 
from utils.eval_metrics import get_global_entity_DIALKG
from tqdm import tqdm
import random

def generate_dataset(data_split,tokenizer,global_ent,test=False,debugging=False):
    data = []
    
    turns_with_kb = 0
    total_turns = 0
    avg_kb_len = []
    for idx_d, dial in tqdm(enumerate(data_split),total=len(data_split)):
        dialogue = []
        history = []
        edge_list = []
        G = nx.Graph()
        for i, d in enumerate(dial):
            total_turns += 1
            
            gold_label = 'gold_KB'
            if 'gold-kb' in d:
                gold_label = 'gold-kb'
                
            if(len(d[gold_label])>0):
                for trip in d[gold_label]:
                    s,r,o = trip
                    G.add_edge(s,o,lable=r)
                edge_list = [tokenizer.encode(' '.join([edge.split("\t")[0],eval(edge.split("\t")[2])['lable'],edge.split("\t")[1]]),add_special_tokens=False) for edge in nx.generate_edgelist(G,delimiter='\t')]
                turns_with_kb += 1    
                avg_kb_len.append(len(G.nodes()))
            if(d['speaker']=="user"):
                history.append(tokenizer.encode(d["text"],add_special_tokens=False))
            else:
                dialogue.append({"history":list(history),
                                "response":tokenizer.encode(d["text"],add_special_tokens=False),
                                "spk":"SYS",
                                "graph": {'edges':edge_list,'adj_mat':None, 'nodes':None}})
                history.append(tokenizer.encode(d["text"],add_special_tokens=False))



        data.append({'id':idx_d,"dialogue":dialogue})
        if(debugging):
            if idx_d == 10: break
    print(f"TURNS with KB: {turns_with_kb/float(total_turns)}")
    print(f"AVG GRAPH NODES: {np.mean(avg_kb_len)}")

    return data

# NOTE:
# kb_percentage is an integer here and it is not actually kb percentage, 
# but instead it is the number of iterations to take from the generation process
# number of iteration is the number of row from the ./data/opendialkg/generation_iteration.csv
def load_DIALKG(args,tokenizer,test_flag=False,kb_percentage=0,debugging=False):
    if(test_flag):
        test =  generate_dataset(json.load(open(f'{args.dataset_path}/opendialkg/test.json')),tokenizer,debugging)
        return None, None, test, None
    else:
        train = generate_dataset(json.load(open(f'{args.dataset_path}/opendialkg/train.json')),tokenizer,debugging)
        dev =   generate_dataset(json.load(open(f'{args.dataset_path}/opendialkg/validation.json')),tokenizer,debugging)
        test =  generate_dataset(json.load(open(f'{args.dataset_path}/opendialkg/test.json')),tokenizer,debugging)
        data = {"train":train,"valid":dev, "test":test}
        
        print('Len train set: ', len(train))
        print('Len dev set: ', len(dev))
        print('Len test set: ', len(test))
        
        # Augment Knowledge based on number of iteration in kb_percentage
        if kb_percentage > 0:
            # Load augmentation data
            gen_dialogues = []
            for gen_dialogue_file in gen_dialogue_files:
                gen_dialogues += json.load(open(f'{args.dataset_path}/generated_dialogue_bs100_rs{random_seed}.json','r'))    
            random.seed(0)
            augment_data = random.sample(gen_dialogues, kb_percentage)
            augment = generate_dataset(augment_data,tokenizer,debugging)
            
            train += augment
            
        print('Len Train augmented: ',len(train))
        
        train_loader, valid_loader, test_loader = get_loader(args, data, tokenizer)
        print(f"Max Len:{test_dataloader(args,train_loader)}")
        print(f"Max Len:{test_dataloader(args,valid_loader)}")
        print(f"Max Len:{test_dataloader(args,test_loader)}")
        return train_loader, valid_loader, test_loader
    
