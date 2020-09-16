import json
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import collections
import torch
from collections import defaultdict
from utils.hugging_face import SPECIAL_TOKENS,MODEL_INPUTS, PADDED_INPUTS, PADDED_SPECIAL, get_loader, build_input_from_segments, test_dataloader
from tabulate import tabulate
from tqdm import tqdm
import pickle
import os.path
import re

def get_dialogue(dial,tokenizer):
    dialogue = []
    history = []
    for i, d in enumerate(dial):
        if(d['spk']=='USR' or d['spk']=='API'):
            history.append(tokenizer.encode(d["text"],add_special_tokens=False))
        else:
            dialogue.append({"history":list(history),
                             "response":tokenizer.encode(d["text"],add_special_tokens=False),
                             "spk":d['spk']})
            history.append(tokenizer.encode(d["text"],add_special_tokens=False))
    return dialogue


def generate_dataset(data_split,tokenizer,debugging,domain=None):
    data = []
    number_of_dialogues = 0
    for _,v in tqdm(data_split.items(), total=len(data_split)):
        dialogue = get_dialogue(v['conversation'],tokenizer)
        data.append({'id':v["src"].lower(),'edges':[],'adj_mat':[],'nodes':[], "dialogue":dialogue, "domain":domain})
        number_of_dialogues += 1
        if(debugging and number_of_dialogues >10): break
    return data


def load_MWOZ(args,tokenizer,test_flag=False,debugging=False):
    train = json.load(open("data/MultiWOZ_2.1/train_data.json"))
    valid = json.load(open("data/MultiWOZ_2.1/valid_data.json"))
    test = json.load(open("data/MultiWOZ_2.1/test_data.json"))
    if(test_flag):
        test = generate_dataset(test,tokenizer,debugging=debugging)
        return None, None, test, None
    else:
        train = generate_dataset(train,tokenizer,debugging=debugging)
        dev = generate_dataset(valid,tokenizer,debugging=debugging)
        test = generate_dataset(test,tokenizer,debugging=debugging)
        dataset_dict = {"train":train,"valid":dev,"test":test}
        train_loader, valid_loader, test_loader = get_loader(args, dataset_dict, tokenizer)
        print(f"Max Len:{test_dataloader(args,train_loader)}")
        print(f"Max Len:{test_dataloader(args,valid_loader)}")
        print(f"Max Len:{test_dataloader(args,test_loader)}")
        return train_loader, valid_loader, None


def load_MWOZ_SINGLE(args,tokenizer,test_flag=False,debugging=False,kb_percentage=0):
    if(test_flag):
        test = []
        for d in ["train","hotel","attraction","restaurant","taxi"]:
            test += generate_dataset(json.load(open(f"data/MultiWOZ_2.1/test/{d}_single.json")),tokenizer,debugging=debugging, domain=d)
        return None, None, test, None
    else:
        train = []
        for d in ["train","hotel","attraction","restaurant","taxi"]:
            train += generate_dataset(json.load(open(f"data/MultiWOZ_2.1/train/{d}_single.json")),tokenizer,debugging=debugging, domain=d)
            if (d == "taxi" and args.up_sampler): # double taxi training data
                train += generate_dataset(json.load(open(f"data/MultiWOZ_2.1/train/{d}_single.json")),tokenizer,debugging=debugging, domain=d)
                # train += generate_dataset(json.load(open(f"data/MultiWOZ_2.1/train/{d}_single.json")),tokenizer,debugging=debugging, domain=d)
            
            if (kb_percentage>0 and d != "taxi"):
                train += generate_dataset(json.load(open(f"data/MultiWOZ_2.1/train/{d}_augmented_{kb_percentage}_single.json")),tokenizer,debugging=debugging, domain=d)
                
                if args.up_sampler: # triple attraction and hotel augmented data
                    if d == "attraction" or d == "hotel":
                        train += generate_dataset(json.load(open(f"data/MultiWOZ_2.1/train/{d}_augmented_{kb_percentage}_single.json")),tokenizer,debugging=debugging, domain=d)
                        train += generate_dataset(json.load(open(f"data/MultiWOZ_2.1/train/{d}_augmented_{kb_percentage}_single.json")),tokenizer,debugging=debugging, domain=d)
                        
                # to make the distribution balance
                #train += generate_dataset(json.load(open(f"data/MultiWOZ_2.1/train/taxi_single.json")),tokenizer,debugging=debugging)

        valid = []
        for d in ["train","hotel","attraction","restaurant","taxi"]:
            valid += generate_dataset(json.load(open(f"data/MultiWOZ_2.1/valid/{d}_single.json")),tokenizer,debugging=debugging, domain=d)

        test = []
        for d in ["train","hotel","attraction","restaurant","taxi"]:
            test += generate_dataset(json.load(open(f"data/MultiWOZ_2.1/test/{d}_single.json")),tokenizer,debugging=debugging, domain=d)
        dataset_dict = {"train":train,"valid":valid,"test":test}
        train_loader, valid_loader, test_loader = get_loader(args, dataset_dict, tokenizer)
        print(f"Max Len:{test_dataloader(args,train_loader)}")
        print(f"Max Len:{test_dataloader(args,valid_loader)}")
        print(f"Max Len:{test_dataloader(args,test_loader)}")
        return train_loader, valid_loader, None