import json
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import collections
from copy import deepcopy

def get_type_dict(kb_path): 
    """
    Specifically, we augment the vocabulary with some special words, one for each of the KB entity types 
    For each type, the corresponding type word is added to the candidate representation if a word is found that appears 
    1) as a KB entity of that type, 
    """
    type_dict = {"address":set(),
                 "area": set(),
                 "food": set(),
                 "location": set(),
                 "phone": set(),
                 "pricerange": set(),
                 "postcode": set(),
                 "type": set(),
                 "id": set(),
                 "name": set()}

    fd = json.load(open(kb_path)) 

    for line in fd:
        for k, v in line.items():
            if(k =="postcode"):
                type_dict[k].add(v.replace(".","").replace(",","").replace(" ","").lower())   
            else:
                type_dict[k].add(v.replace(" ","_").lower())   
    return type_dict

def entityList(kb_path): 
    KB = json.load(open(kb_path))
    glob = set()
    for item in KB:
        for k, v in item.items():
            if(k =="postcode"):
                glob.add(v.replace(".","").replace(",","").replace(" ","").lower())   
            else: 
                glob.add(v.replace(" ","_").lower())
    glob = list(glob)
    return glob

def load_entity(path):
    global_ent = entityList(path)
    type_dict = get_type_dict(path)
    # print(type_dict.keys())
    # print("COUSI",type_dict['R_cuisine'])
    return global_ent, type_dict

def delexicalize_CamRest(global_entity, sentence, type_dict, rec_delex=False, past_type_record={}, max_cnt=0):
    sketch_response = []
    type_record = deepcopy(past_type_record) # key: entity_type, value: dict

    entities = ["_phone","_cuisine","_address","_location","_number","_price","_rating"]
    entities = entities + ["_post_code"] # for dstc2 task6

    if "moderately" in sentence: 
        sentence = sentence.replace("moderately","moderate ly")

    if "expensively" in sentence: 
        sentence = sentence.replace("expensively","expensive ly")
    
    words = sentence.split()
    for i in range(len(words)):
        word = words[i]
        if (word in global_entity and word != "ask") or (i < len(words)-1 and word == "ask" and words[i+1] == 'is'):
            ent_type = None
            for kb_item in type_dict.keys():
                if word in type_dict[kb_item]:
                    ent_type = kb_item
                    break

            # special case
            is_special_case = False
            for ent in entities:
                if ent in word:
                    word = word.replace(ent,"")
                    is_special_case = True
                    break
            
            if rec_delex:
                if ent_type in type_record:
                    if word not in type_record[ent_type]:
                        type_record[ent_type][word] = len(type_record[ent_type]) + 1
                else:
                    type_record[ent_type] = {}
                    type_record[ent_type][word] = 1
                
                # find the index of the entity
                if is_special_case:
                    # if there is no api call
                    if "R_restaurant" not in type_record:
                        type_record["R_restaurant"] = {}
                    if word not in type_record["R_restaurant"]:
                        type_record["R_restaurant"][word] = 1

                    count = type_record["R_restaurant"][word]
                else:
                    count = type_record[ent_type][word]

                # special case for babi task5
                if type_record["api call"] == 1:
                    count = count + max_cnt  # the index starts at 3 after the api call
                else:
                    max_cnt = max(max_cnt,count)
                
                if i+1 < len(words) and words[i+1] == "ly":
                    words[i+1] = ""
                    sketch_response.append('@'+ent_type+'_'+str(count)+"ly")
                else:
                    sketch_response.append('@'+ent_type+'_'+str(count))
            else:
                if i+1 < len(words) and words[i+1] == "ly":
                    words[i+1] = ""
                    sketch_response.append('@'+ent_type+"ly")
                else:
                    sketch_response.append('@'+ent_type)
                    
        else:
            # special case for dstc2
            is_special_case = False
            for ent in entities:
                if "R" + ent in word:
                    ent_type = "R" + ent
                    if ent_type in type_record:
                        if word not in type_record[ent_type]:
                            count = len(type_record[ent_type]) + 1
                    else:
                        type_record[ent_type] = {}
                        count = 1

                    is_special_case = True
                    break

            if is_special_case:
                if rec_delex:
                    sketch_response.append('@'+ent_type+'_'+str(count))
                else:
                    sketch_response.append('@'+ent_type)
            else:
                sketch_response.append(word)
    sketch_response = " ".join(sketch_response).replace("  ", " ")
    return sketch_response, type_record, max_cnt

def preprocess_data(global_ent, sentence):
    for ent in global_ent:
        ent_no_underscore = ent.replace('_', ' ')
        if ent_no_underscore in sentence:
            sentence = sentence.replace(ent_no_underscore, ent)

    return sentence

def generate_CAMREST_template(file_path, kb_file_path, rec_delex=False, verbose=False):
    conversation = []
    type_record = {} # key: entity_type, value: dict
    type_record["api call"] = 0 # special case for babi task 5

    global_ent, type_dict = load_entity(kb_file_path)

    # collect all data and delexicalize the sequence if mentioned
    with open(file_path,'r') as f:
        for line in f:
            if line == "\n": continue
            nid, line = line.replace("\n","").split(' ', 1)
            if nid == "1": 
                type_record = {}
                type_record["api call"] = 0 # special case for babi task 5
                max_cnt = 0 

            if ("\t" in line):
                if verbose:
                    print(f"LINE >>> {line}")
                usr_res, sys_res = line.split("\t")
                if usr_res.strip() == "":
                    usr_res = "<SILENCE>"

                # if task_id == 6: # special case for dstc2, babi task 6
                #     sys_res = preprocess_data(global_ent, sys_res)

                usr_delex, type_record, max_cnt = delexicalize_CamRest(global_ent, usr_res, type_dict, rec_delex=rec_delex, past_type_record=type_record, max_cnt=max_cnt)
                sys_delex, type_record, max_cnt = delexicalize_CamRest(global_ent, sys_res, type_dict, rec_delex=rec_delex, past_type_record=type_record, max_cnt=max_cnt)
                conversation.append((nid, usr_delex, sys_delex))

                if sys_res.split()[0] == "api_call": 
                    type_record = {}
                    type_record["api call"] = 1
                    

                if verbose:
                    print(f"USR >>> {usr_res}")
                    print(f"USR_TEMP >>> {usr_delex}")
                    print(f"SYS >>> {sys_res}")
                    print(f"SYS_TEMP >>> {sys_delex}")
                    print()

    num_conversation, unique_conversation, temp_conversation = 0, {}, []
    unique_sentences = {}

    out_file_path = file_path.replace(".txt", "")
    if rec_delex:
        out_file_path += "_record-delex"
    else:
        out_file_path += "_delex"

    # with open(out_file_path + "_template.txt", "w+") as f_out_template:
    with open(out_file_path + ".txt", "w+") as f_out:
        print("Reading: {}".format(file_path))

        for i in range(len(conversation)):
            turn = conversation[i]

            if turn[0] == "1": 
                if i > 0: 
                    f_out.write("\n")

                    # check if the dialogue is unique
                    key = " ".join(t[1] + " " + t[2] for t in temp_conversation)
                    # if key not in unique_conversation:
                    #     for conv in temp_conversation:
                    #         f_out_template.write("{} {}\t{}\n".format(conv[0], conv[1], conv[2]))
                    #     f_out_template.write("\n")
                    unique_conversation[key] = True

                    temp_conversation = []
                    num_conversation += 1
            
            temp_conversation.append((turn[0], turn[1], turn[2]))
            f_out.write("{} {}\t{}\n".format(turn[0], turn[1], turn[2]))
            unique_sentences[(turn[1], turn[2])] = True
            
            if i == len(conversation)-1 and temp_conversation != "": 
                # check if the dialogue is unique
                key = " ".join(t[1] + " " + t[2] for t in temp_conversation)
                # if key not in unique_conversation:
                #     for conv in temp_conversation:
                #         f_out_template.write("{} {}\t{}\n".format(conv[0], conv[1], conv[2]))
                #     f_out_template.write("\n")
                unique_conversation[key] = True

                num_conversation += 1

    print("Number of convs: {} unique convs: {} unique sents: {}".format(num_conversation, len(unique_conversation), len(unique_sentences)))

files = ['CamRest/train.txt']
for file_path in files:
    print("> Delexicalization")
    generate_CAMREST_template(file_path, kb_file_path='CamRest/KB.json', rec_delex=False)
    print("> Recorded Delexicalization")
    generate_CAMREST_template(file_path, kb_file_path='CamRest/KB.json', rec_delex=True)
    print("")