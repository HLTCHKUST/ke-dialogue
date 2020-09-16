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
from tqdm import tqdm

def get_dialogue(dial,tokenizer):
    dialogue = []
    history = []
    for _, d in enumerate(dial):
        if(d['spk']=='USR'):
            history.append(tokenizer.encode(d["text"],add_special_tokens=False))
        else:
            dialogue.append({"history":list(history),
                             "response":tokenizer.encode(d["text"],add_special_tokens=False),
                             "spk":d['spk']})
            history.append(tokenizer.encode(d["text"],add_special_tokens=False))
    return dialogue

def generate_dataset(data_split,tokenizer,debugging=False,edges=False):
    num_lines = sum(1 for line in open(data_split,'r'))
    with open(data_split,'r') as f:
        conversation = []
        data = []
        KB = []
        idd = 0
        for line in tqdm(f,total=num_lines):
            if(line == "\n"):
                # for c in conversation:
                #     print(f"{c['spk']} >>> {c['text']}")
                # print()
                # print()
                dialogue = get_dialogue(conversation,tokenizer)
                if(edges):
                    KB = [tokenizer.encode(" ".join(k),add_special_tokens=False) for k in KB]
                else:
                    KB = []
                data.append({'id':idd,"dialogue":dialogue,"edges":KB})
                idd += 1
                conversation = []
                KB = []
            else:
                _, line = line.replace("\n","").split(' ', 1)
                if ("\t" in line):
                
                    user, syst = line.split("\t")
                    if(edges):
                        # print(user)
                        conversation.append({"spk":"USR","text":user})
                        conversation.append({"spk":"SYS","text":syst})
                    else:
                        if("<SILENCE>" not in user):
                            conversation.append({"spk":"USR","text":user})
                        if("i'm on it" not in syst and "api_call" not in syst and "ok let me look into some options for you" not in syst):
                            conversation.append({"spk":"SYS","text":syst})
                else:
                     KB.append(line.split())

    return data


def load_CAMREST(args,tokenizer,test_flag=False,debugging=False,kb_percentage=0):
    if(test_flag):
        test = generate_dataset('data/CamRest/test.txt',tokenizer,debugging=debugging,edges=args.flatten_KB)
        return None, None, test
    else:

        train = generate_dataset('data/CamRest/train.txt',tokenizer,debugging=debugging,edges=args.flatten_KB)
        if(kb_percentage>0):
            train += generate_dataset(f'data/CamRest/gen-babi7-nk201-nd{kb_percentage}-rs0.txt',tokenizer,debugging=debugging,edges=args.flatten_KB)
        dev = generate_dataset('data/CamRest/dev.txt',tokenizer,debugging=debugging,edges=args.flatten_KB)
        test = generate_dataset('data/CamRest/test.txt',tokenizer,debugging=debugging,edges=args.flatten_KB)
        smd = {"train":train,"valid":dev, "test":test}
        train_loader, valid_loader, test_loader = get_loader(args, smd, tokenizer)
        print(f"Max Len:{test_dataloader(args,train_loader)}")
        print(f"Max Len:{test_dataloader(args,valid_loader)}")
        print(f"Max Len:{test_dataloader(args,test_loader)}")
        return train_loader, valid_loader, test_loader


