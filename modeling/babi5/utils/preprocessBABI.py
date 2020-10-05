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

def generate_dataset(data_split,tokenizer,debugging=False):
    num_lines = sum(1 for line in open(data_split,'r'))
    with open(data_split,'r') as f:
        conversation = []
        data = []
        idd = 0
        for line in tqdm(f,total=num_lines):
            if(line == "\n"):
                # for c in conversation:
                #     print(f"{c['spk']} >>> {c['text']}")
                # print()
                # print()
                dialogue = get_dialogue(conversation,tokenizer)
                data.append({'id':idd,"dialogue":dialogue})
                idd += 1
                conversation = []
            else:
                _, line = line.replace("\n","").split(' ', 1)
                if ("\t" in line):
                    user, syst = line.split("\t")
                    if("<SILENCE>" not in user):
                        conversation.append({"spk":"USR","text":user})
                    if("i'm on it" not in syst and "api_call" not in syst and "ok let me look into some options for you" not in syst):
                        conversation.append({"spk":"SYS","text":syst})

    return data


def load_BABI(args,tokenizer,test_flag=False,OOV=False,debugging=False,kb_percentage=0):
    if(test_flag):
        if(OOV):
            test = generate_dataset(f'{args.dataset_path}/dialog-babi-task5tst-OOV.txt',tokenizer,debugging=debugging)
        else:
            test = generate_dataset(f'{args.dataset_path}/dialog-babi-task5tst.txt',tokenizer,debugging=debugging)
        return None, None, test
    else:

        train = generate_dataset(f'{args.dataset_path}/dialog-babi-task5trn.txt',tokenizer,debugging=debugging)
        if(kb_percentage>0):
            train += generate_dataset(f'{args.dataset_path}/gen-babi5-nk558-nd{kb_percentage}-rs0.txt',tokenizer,debugging=debugging)
        dev = generate_dataset(f'{args.dataset_path}/dialog-babi-task5dev.txt',tokenizer,debugging=debugging)
        test = generate_dataset(f'{args.dataset_path}/dialog-babi-task5tst.txt',tokenizer,debugging=debugging)
        smd = {"train":train,"valid":dev, "test":test}
        train_loader, valid_loader, test_loader = get_loader(args, smd, tokenizer)
        print(f"Max Len:{test_dataloader(args,train_loader)}")
        print(f"Max Len:{test_dataloader(args,valid_loader)}")
        print(f"Max Len:{test_dataloader(args,test_loader)}")
        return train_loader, valid_loader, test_loader

    
def load_DSTC2(args,tokenizer,test_flag=False,OOV=False,debugging=False):
    if(test_flag):
        test = generate_dataset('data/dialog-bAbI-tasks/dialog-babi-task6tst.txt',tokenizer,debugging=debugging)
        return None, None, test
    else:
        train = generate_dataset('data/dialog-bAbI-tasks/dialog-babi-task6trn.txt',tokenizer,debugging=debugging)
        dev = generate_dataset('data/dialog-bAbI-tasks/dialog-babi-task6dev.txt',tokenizer,debugging=debugging)
        test = generate_dataset('data/dialog-bAbI-tasks/dialog-babi-task6tst.txt',tokenizer,debugging=debugging)
        smd = {"train":train,"valid":dev, "test":test}
        train_loader, valid_loader, test_loader = get_loader(args, smd, tokenizer)
        print(f"Max Len:{test_dataloader(args,train_loader)}")
        print(f"Max Len:{test_dataloader(args,valid_loader)}")
        print(f"Max Len:{test_dataloader(args,test_loader)}")
        return train_loader, valid_loader, test_loader
