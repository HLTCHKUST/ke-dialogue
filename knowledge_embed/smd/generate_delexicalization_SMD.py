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
    type_dict = {
                 "event": set(),
                 "time": set(),
                 "date": set(),
                 "party": set(),
                 "location" : set(),
                 "room": set(),
                 "agenda": set(),
                 "weekly_time": set(),
                 "temperature": set(),
                 "weather_attribute": set(),
                 "traffic_info": set(),
                 "poi_type": set(),
                 "poi": set(),
                 "poi_address": set(),
                 "distance": set()
                 }


    fd = json.load(open(kb_path)) 
    for k,val in fd.items():
        if(k != 'poi'):
            for v in val:
                type_dict[k].add(v.lower().replace(' ', '_'))   
        else:
            for v in val:
                type_dict["poi"].add(v['poi'].replace(" ","_").lower())   
                type_dict["poi_address"].add(v['address'].replace(" ","_").lower())   
    return type_dict


def entityList(kb_path): 
    with open(kb_path) as f:
        global_entity = json.load(f)
        global_entity_list = []
        for key in global_entity.keys():
            if key != 'poi':
                global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
            else:
                for item in global_entity['poi']:
                    global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
        global_entity_list = list(set(global_entity_list))
    return global_entity_list

def load_entity(path):
    global_ent = entityList(path)
    type_dict = get_type_dict(path)
    # print(type_dict.keys())
    # print("COUSI",type_dict['R_cuisine'])
    return global_ent, type_dict

def delexicalize_SMD(global_entity, sentence, type_dict, rec_delex=False, past_type_record={},KB_DICT={}):
    sketch_response = []
    type_record = deepcopy(past_type_record) # key: entity_type, value: dict

    entities = ["_phone","_cuisine","_address","_location","_number","_price","_rating"]
    entities = entities + ["_post_code"] # for dstc2 task6
    
    words = sentence.split()
    for i in range(len(words)):
        word = words[i]
        if (word in global_entity):
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
                
                sketch_response.append('@'+ent_type+'_'+str(count))
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
    return sketch_response, type_record

def preprocess_data(global_ent, sentence):
    for ent in global_ent:
        ent_no_underscore = ent.replace('_', ' ')
        if ent_no_underscore in sentence:
            sentence = sentence.replace(ent_no_underscore, ent)

    return sentence

def generate_SMD_template(file_path, kb_file_path, rec_delex=False, verbose=False):
    conversation = []
    KB = []
    type_record = {} # key: entity_type, value: dict
    type_record["api call"] = 0 # special case for babi task 5

    global_ent, type_dict = load_entity(kb_file_path)

    # collect all data and delexicalize the sequence if mentioned
    with open(file_path,'r') as f:
        for line in f:
            if line == "\n": continue
            if '#' in line:
                line = line.replace("#","")
                task_type = line.replace("\n","")
                type_record = {} # key: entity_type, value: dict
                KB = []
                continue
            nid, line = line.replace("\n","").split(' ', 1)

            if ("\t" in line):
                if verbose:
                    print(f"LINE >>> {line}")
                usr_res, sys_res, _ = line.split("\t")
                if usr_res.strip() == "":
                    usr_res = "<SILENCE>"

                KB_DICT = {}
                if(task_type=="weather"):
                    for k in KB:
                        if(len(k)==7):
                            if k[0] not in KB_DICT:
                                KB_DICT[k[0]] = {"monday": [],"tuesday": [], "friday": [], "wednesday": [], "thursday": [], "sunday": [], "location": [], "saturday": []}
                            KB_DICT[k[0]][k[1]] = {"w":k[2],"low":k[4],"high":k[6]}
                usr_delex, type_record = delexicalize_SMD(global_ent, usr_res, type_dict, rec_delex=rec_delex, past_type_record=type_record,KB_DICT=KB_DICT)
                sys_delex, type_record = delexicalize_SMD(global_ent, sys_res, type_dict, rec_delex=rec_delex, past_type_record=type_record,KB_DICT=KB_DICT)
                conversation.append((nid, usr_delex, sys_delex))

                if verbose:
                    print(f"USR >>> {usr_res}")
                    print(f"USR_TEMP >>> {usr_delex}")
                    print(f"SYS >>> {sys_res}")
                    print(f"SYS_TEMP >>> {sys_delex}")
                    print()
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

files = ['SMD/train.txt']
for file_path in files:
    print("> Delexicalization")
    generate_SMD_template(file_path, kb_file_path='SMD/kvret_entities.json', rec_delex=False)
    print("> Recorded Delexicalization")
    generate_SMD_template(file_path, kb_file_path='SMD/kvret_entities.json', rec_delex=True)
    print("")