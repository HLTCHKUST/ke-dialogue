import json
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import collections
import torch
from collections import defaultdict
from hugging_face import get_loader, test_dataloader
import ast
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
        KB = []
        idd = 0
        for line in tqdm(f,total=num_lines):
            line = line.strip()
            if line:
                if '#' in line:
                    if(idd!=0):
                        dialogue = get_dialogue(conversation,tokenizer)
                        KB = [tokenizer.encode(" ".join(k),add_special_tokens=False) for k in KB]
                        data.append({'id':idd,"domain":task_type,"dialogue":dialogue, "edges":KB})
                    idd += 1
                    conversation = []
                    KB = []
                    line = line.replace("#","")
                    task_type = line
                    continue

                _, line = line.split(' ', 1)
                if '\t' in line:
                    u, r, _ = line.split('\t')
                    conversation.append({"spk":"USR","text":u})
                    conversation.append({"spk":"SYS","text":r})
                else:
                    if(len(line.split())==5 and task_type=="navigate"): 
                        KB.append(line.split())
                    elif(task_type=="weather"):
                        if(len(line.split())==3):
                            KB.append(line.split())
                        elif(len(line.split())==4):
                            KB[-1] += [line.split()[-2],line.split()[-1]]
                    else:
                        KB.append(line.split()) 
    return data 

def generate_dataset_FINETUNE(data_split,tokenizer,debugging=False):
    num_lines = sum(1 for line in open(data_split,'r'))
    with open(data_split,'r') as f:
        conversation = []
        data = []
        idd = 0
        for line in tqdm(f,total=num_lines):
            line = line.strip()
            if line:
                _, line = line.split(' ', 1)
                if '\t' in line:
                    u, r = line.split('\t')
                    conversation.append({"spk":"USR","text":u})
                    conversation.append({"spk":"SYS","text":r})
            else:
                dialogue = get_dialogue(conversation,tokenizer)
                data.append({'id':idd,"dialogue":dialogue,"domain":None})
                idd += 1
                conversation = []
    return data 


def load_SMD(args,tokenizer,test_flag=False,debugging=False,delex=False):    
    if(test_flag):
        test = generate_dataset("data/SMD/test.txt",tokenizer,debugging)
        return test, None
    else:
        train = generate_dataset("data/SMD/train.txt",tokenizer,debugging)
        dev = generate_dataset("data/SMD/dev.txt",tokenizer,debugging)
        test = generate_dataset("data/SMD/test.txt",tokenizer,debugging)
        smd = {"train":train,"valid":dev, "test":test}
        train_loader, valid_loader, test_loader = get_loader(args, smd, tokenizer)
        print(f"Max Len:{test_dataloader(args,train_loader)}")
        print(f"Max Len:{test_dataloader(args,valid_loader)}")
        print(f"Max Len:{test_dataloader(args,test_loader)}")
        return train_loader, valid_loader, test_loader


def generate_template(global_entity, sentence, sent_ent, kb_arr, domain):
    """
    Based on the system response and the provided entity table, the output is the sketch response. 
    """
    # print(sentence, sent_ent, kb_arr, domain)
    
    sketch_response = [] 
    counter = defaultdict(list)
    if sent_ent == []:
        sketch_response = sentence.split()
    else:
        for word in sentence.split():
            if word not in sent_ent:
                sketch_response.append(word)
            else: 
                for key in global_entity.keys():
                    if key!='poi':
                        global_entity[key] = [x.lower() for x in global_entity[key]]
                        if word in global_entity[key]:
                            if word not in counter[key]:
                                counter[key].append(word)
                            ent_type = key+"_"+str(counter[key].index(word))
                            break
                        elif word.replace('_', ' ') in global_entity[key]:
                            if word not in counter[key]:
                                counter[key].append(word)
                            ent_type = key+"_"+str(counter[key].index(word))
                            break
                    else:
                        poi_list = [d['poi'].lower() for d in global_entity['poi']]
                        if word in poi_list:
                            if word not in counter[key]:
                                counter[key].append(word)
                            ent_type = key+"_"+str(counter[key].index(word))
                            break
                        elif word.replace('_', ' ') in poi_list:
                            if word not in counter[key]:
                                counter[key].append(word)
                            ent_type = key+"_"+str(counter[key].index(word))
                            break
                        
                        address_list = [d['address'].lower() for d in global_entity['poi']]
                        if word in address_list:
                            if word not in counter['poi_address']:
                                counter['poi_address'].append(word)
                            ent_type = "poi_address_"+str(counter['poi_address'].index(word))
                            break
                        elif word.replace('_', ' ') in address_list:
                            if word not in counter['poi_address']:
                                counter['poi_address'].append(word)
                            ent_type = "poi_address_"+str(counter['poi_address'].index(word))
                            break

                if ent_type == None:
                    print(sentence, sent_ent, kb_arr, domain)
                sketch_response.append("@"+ent_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response
        
def delex_SMD(file_name, global_entity, max_line = None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr = [], [], [], []
    max_resp_len = 0
    
    conversation = []
    with open(file_name) as fin:
        cnt_lin, sample_counter = 1, 1
        for line in fin:
            line = line.strip()
            if line:
                if '#' in line:
                    line = line.replace("#","")
                    task_type = line
                    continue

                nid, line = line.split(' ', 1)
                if '\t' in line:
                    u, r, gold_ent = line.split('\t')
            
                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)

                    # find gold ent for user.
                    u_gold_ent = [t for t in u.split(" ") if "_" in t] + gold_ent
                    # print(u_gold_ent, gold_ent)

                    ent_idx_cal, ent_idx_nav, ent_idx_wet = [], [], []
                    if task_type == "weather": ent_idx_wet = gold_ent
                    elif task_type == "schedule": ent_idx_cal = gold_ent
                    elif task_type == "navigate": ent_idx_nav = gold_ent
                    ent_index = list(set(ent_idx_cal + ent_idx_nav + ent_idx_wet))
                    
                    usr_delex = generate_template(global_entity, u, u_gold_ent, kb_arr, task_type)
                    sys_delex = generate_template(global_entity, r, gold_ent, kb_arr, task_type)

                    conversation.append((nid, usr_delex, sys_delex))
    

    num_conversation, unique_conversation, temp_conversation = 0, {}, []
    unique_sentences = {}

    out_file_path = file_name.replace(".txt", "_delex.txt")

    with open(out_file_path, "w+") as f_out:
        print("Saving to: {}".format(out_file_path))

        for i in range(len(conversation)):
            turn = conversation[i]

            if turn[0] == "1": 
                if i > 0: 
                    f_out.write("\n")

                    # check if the dialogue is unique
                    key = " ".join(t[1] + " " + t[2] for t in temp_conversation)
                    unique_conversation[key] = True

                    temp_conversation = []
                    num_conversation += 1
            
            temp_conversation.append((turn[0], turn[1], turn[2]))
            f_out.write("{} {}\t{}\n".format(turn[0], turn[1], turn[2]))
            unique_sentences[(turn[1], turn[2])] = True
            
            if i == len(conversation)-1 and temp_conversation != "": 
                # check if the dialogue is unique
                key = " ".join(t[1] + " " + t[2] for t in temp_conversation)
                unique_conversation[key] = True

                num_conversation += 1

    print("Number of convs: {} unique convs: {} unique sents: {}".format(num_conversation, len(unique_conversation), len(unique_sentences)))

                    
    
if __name__ == "__main__":
    global_entity = json.load(open("SMD/kvret_entities.json"))

    delex_SMD("SMD/train.txt", global_entity)
    delex_SMD("SMD/dev.txt", global_entity)
    delex_SMD("SMD/test.txt", global_entity)

    print("Yay")