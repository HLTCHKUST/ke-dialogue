import os,sys
import argparse
import pandas as pd
import re
import math
import timeit
import copy
import json
import random
from sklearn.utils import shuffle
from tqdm import tqdm
import json
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import collections
import pprint
from tabulate import tabulate

import sqlite3
from collections import defaultdict  

from revertible_string import RevertibleString
from generate_SMD_sql import generate_schedule_sql, generate_navigate_sql, generate_weather_sql

pp = pprint.PrettyPrinter(indent=4, width=200)

# To distinguish different dialogue template
# filter out the ones that cannot be distinguished
Type2POI = {
    "schedule": "@event",
    "weather": "@location",
    "navigate": "@poi_type",
}

schedule_keys = ['event', 'time', 'date', 'room', 'agenda', 'party']
weather_keys = ["location", "date", "weather", "high", "low"]
navigate_keys = ['poi', 'poi_type', 'address', 'distance', 'traffic_info']

Type2Keys = {
    "schedule": ["@event", "@date", "@time", "@party", "@agenda", "@room", "@weekly_time"],
    "weather": ["@location", "@date", "@weather_attribute", "@temperature_high", "@temperature_low", "@weekly_time", "@today"],
    "navigate": ["@poi", "@poi_type", "@distance", "@traffic_info", "@poi_address", "@party", "@weekly_time"],
    "NOTYPE": None,
}

Traffic2Num = {
    'road block nearby': 4,
    'car collision nearby': 3, 
    'heavy traffic': 2, 
    'moderate traffic': 1, 
    'no traffic': 0,
}

Num2Traffic = {
    0: 'no_traffic',
    1: 'moderate_traffic',
    2: 'heavy_traffic',
    3: 'car_collision_nearby',
    4: 'road_block_nearby',
}

Date2Num = {
    'monday': '1',
    'tuesday': '2', 
    'wednesday': '3', 
    'thursday': '4', 
    'friday': '5',
    'saturday': '6',
    'sunday': '7',
}

Type2NumKB = {
    "train": {"schedule": 415, "weather": 797, "navigate": 800,},
    "dev": {"schedule": 100, "weather": 99, "navigate": 100,},
    "test": {"schedule": 100, "weather": 100, "navigate": 100,},
}

DIST = {
    "schedule": {},
    "weather": {},
    "navigate": {},
}

def Key2Idx(dialog_type, key):
    if dialog_type in ["navigate", "location information"]:
        index = navigate_keys.index(key)
    elif dialog_type in ["weather", "weekly forecast"]:
        index = weather_keys.index(key)
    elif dialog_type in ["schedule", "calendar"]:
        index = schedule_keys.index(key)
    return index

def build_db(kb_folder, split):
    rain_record_map = {}

    with open("SMD/kvret_test_public.json", "r") as f:
        data = json.load(f)

    for i, item in enumerate(data):
        keys = item["scenario"]["kb"]["column_names"]
        content = item["scenario"]["kb"]["items"]
        dialog_type = item["scenario"]["kb"]["kb_title"]
        if content is None:
            continue
        
        if os.path.exists(f'{kb_folder}/{split}/dialog_{i}.db'):
            os.remove(f'{kb_folder}/{split}/dialog_{i}.db')
        conn = sqlite3.connect(f'{kb_folder}/{split}/dialog_{i}.db')
        c = conn.cursor()
        if dialog_type == "weekly forecast":
            c.execute("CREATE TABLE content (location, date, weather, high INTEGER, low INTEGER)")
        elif dialog_type == "location information":
            c.execute("CREATE TABLE content (poi, poi_type, address, distance INTEGER, traffic_info INTEGER)")
        elif dialog_type == "calendar":
            c.execute("CREATE TABLE content (event, time, date, room, agenda, party)")
        
        if dialog_type == "calendar":
            for j, tup in enumerate(content):
                values = {n:"-" for n in keys}
                temp = ','.join(["?" for _ in range(len(keys))])
                for key_tup, val_tup in tup.items():
                    values[key_tup] = val_tup
                c.execute(f"insert into content values ({temp})",[str(h) for h in values.values()])
                conn.commit()
        elif dialog_type == "location information":
            for j, tup in enumerate(content):
                values = {n:"-" for n in keys}
                temp = ','.join(["?" for _ in range(len(keys))])
                for key_tup, val_tup in tup.items():
                    if key_tup == "distance":
                        values[key_tup] = int(val_tup[:-6])
                    elif key_tup == "traffic_info":
                        values[key_tup] = Traffic2Num[val_tup]
                    else:
                        values[key_tup] = val_tup
                c.execute(f"insert into content values ({temp})",[str(h) for h in values.values()])
                conn.commit()
        elif dialog_type == "weekly forecast":
            keys = weather_keys
            for j, tup in enumerate(content):
                values = {n:"-" for n in keys}
                temp = ','.join(["?" for _ in range(len(keys))])
                values["location"] = tup["location"]
                for key_tup, val_tup in tup.items():
                    val_tup = val_tup.split(",")
                    if key_tup not in ["location", "today"]:
                        values["date"] = Date2Num[key_tup] + key_tup
                        if val_tup[0] == "raining":
                            values["weather"] = "rain"
                        else:
                            values["weather"] = val_tup[0]
                        values["low"] = int(val_tup[1][8:-1])
                        values["high"] = int(val_tup[2][8:-1])
                        if val_tup[0] == "raining":
                            if str(i) not in rain_record_map:
                                rain_record_map[f"{i}"] = []
                            rain_record_map[f"{i}"].append((tup["location"], key_tup))
                        c.execute(f"insert into content values ({temp})",[str(h) for h in values.values()])
                conn.commit()
        conn.close()

    with open(f"{kb_folder}/{split}_rain_record.json", "w") as f:
        json.dump(rain_record_map, f)

def read_dialogue(template_path):
    dialogues = []
    dialogue = []
    for line in open(template_path,'r').readlines():
        if len(line) == 1: # Only \n
            dialogues.append(dialogue)
            dialogue = []
        else:
            first_space_index = line.index(' ')
            turn_id = line[:first_space_index]
            conv = line[first_space_index+1:].replace('\n','')
            request, response = conv.split('\t')
            dialogue.append((turn_id, RevertibleString(request), RevertibleString(response)))
    return dialogues

def detect_dialogue_type(meta):
    DETECT = False
    for word in meta:
        if ("@poi_type" in word) and not DETECT:
            DETECT = True
            _type = "navigate"
    for word in meta:
        if ("@weather_attribute" in word or "@temperature" in word) and "@location_1" in meta and not DETECT:
            DETECT = True
            _type = "weather"
    for word in meta:
        if ("@event" in word) and not DETECT:
            DETECT = True
            _type = "schedule"
            
    if not DETECT: 
        _type ="NOTYPE"
    return _type

def generate_metadata(dialogues):
    delexicalized_dialog_meta = [] # Buffer containing list of tuple(dialog, delex_dict, delex_resolved_args_list, dialog_type)
    for dialogue in dialogues:
        delex_to_chat_dict = { } # Dictionary of recoderd delexicalized word to list of Chat object containing the corresponding word
        delex_resolved_args_list = [] # List of all recorded delexicalized words in api_call that need to be resolved for generation

        for turn_id, request, response in dialogue:
            delex_words = request.str.split(" ") + response.str.split(" ")
            for word in delex_words:
                if "@" in word and "_" in word and word not in delex_resolved_args_list:
                    delex_resolved_args_list.append(word) 
        dialog_type = detect_dialogue_type(delex_resolved_args_list)
        if dialog_type not in ["schedule", "weather", "navigate"]:
            continue

        delex_keys = Type2Keys[dialog_type]
        poi = Type2POI[dialog_type]
        
        for turn_id, request, response in dialogue:
            for chat in [request, response]:
                if "@" in chat.str and "_" in chat.str:
                    for delex_key in delex_keys:
                        for i in range(0, 9):
                            if (delex_key == poi):
                                recorded_delex_word = f'{delex_key}_{i}'
                                if recorded_delex_word in chat.str:
                                    if recorded_delex_word not in delex_to_chat_dict:
                                        delex_to_chat_dict[recorded_delex_word] = []
                                    delex_to_chat_dict[recorded_delex_word].append(chat)
                            else:
                                for j in range(0, 9):
                                    recorded_delex_word = f'{delex_key}_{i}{j}'
                                    if recorded_delex_word in chat.str:
                                        if recorded_delex_word not in delex_to_chat_dict:
                                            delex_to_chat_dict[recorded_delex_word] = []
                                        delex_to_chat_dict[recorded_delex_word].append(chat)
        # Add result to global metadata buffer
        delexicalized_dialog_meta.append((dialogue, delex_to_chat_dict, delex_resolved_args_list, dialog_type))
    return delexicalized_dialog_meta

def generate_dialogue_index(dialogues):
    delexicalized_dialog_meta = generate_metadata(dialogues)
    
    meta_data_list = []
    dialog_type_list = []

    for dialogue, delex_to_chat_dict, delex_resolved_args_list, dialog_type in delexicalized_dialog_meta:
        meta_data_list.append((dialogue, delex_to_chat_dict, delex_resolved_args_list, dialog_type))
        dialog_type_list.append(dialog_type)

    index_dialog = pd.DataFrame({'domain':dialog_type_list,'meta':meta_data_list})
    return index_dialog

def dump_dialogue_to_file(str_dialogues, output_file):
    with open(output_file, "w") as f:
        for str_dialogue in str_dialogues:
            for turn_id, request, response in str_dialogue:
                f.write(f'{turn_id} {request}\t{response}\n')
            f.write('\n')

def generate_sql(domain, key_group, constraints):
    if domain == "weather":
        sql, sql_cols = generate_weather_sql(key_group, constraints)
    elif domain == "schedule":
        sql, sql_cols = generate_schedule_sql(key_group, constraints)
    else:
        sql, sql_cols = generate_navigate_sql(key_group, constraints)
    return sql, sql_cols


def get_weather_entities(all_rows, sql_cols, base_resolved_dict):
    for i in range(len(sql_cols)):
        if sql_cols[i] == "weather":
            sql_cols[i] = "weather_attribute"
        elif sql_cols[i] == "low":
            sql_cols[i] = "temperature_low"
        elif sql_cols[i] == "high":
            sql_cols[i] = "temperature_high"
    static_keys = sql_cols[:2] if sql_cols[0] == "weather_attribute_10" else sql_cols[:1]
    loop_keys = sql_cols[2:] if sql_cols[0] == "weather_attribute_10" else sql_cols[1:]
    if len(loop_keys) == 0:
        assert len(all_rows[0]) - len(static_keys) == 0
        loop = 0
    else:
        assert (len(all_rows[0]) - len(static_keys)) % len(loop_keys) == 0 
        loop = int((len(all_rows[0]) - len(static_keys)) / len(loop_keys))
        
    resolved_dicts = []
    
    for row in all_rows:
        resolved_dict = base_resolved_dict.copy()
        
        for idx, key in enumerate(static_keys):
            if key == "location":
                resolved_dict[f"@{key}_1"] = row[idx].replace(" ", "_")
            else:
                resolved_dict[f"@{key}"] = row[idx].replace(" ", "_")
        for idx1 in range(loop):
            for idx2, key in enumerate(loop_keys):
                if "low" in key or "high" in key:
                    resolved_dict[f"@{key}_1{idx1+1}"] = str(row[len(static_keys)+len(loop_keys)*idx1+idx2])+"f"
                    if idx == 0 and idx1 == 0:
                        resolved_dict[f"@{key}_1{idx1}"] = str(row[len(static_keys)+len(loop_keys)*idx1+idx2])+"f"
                elif "date" in key:
                    resolved_dict[f"@{key}_1{idx1+1}"] = row[len(static_keys)+len(loop_keys)*idx1+idx2][1:]
                else:
                    resolved_dict[f"@{key}_1{idx1+1}"] = row[len(static_keys)+len(loop_keys)*idx1+idx2].replace(" ", "_")
        resolved_dicts.append(resolved_dict.copy())    
    return resolved_dicts

def get_navigate_entities(all_rows, sql_cols, base_resolved_dict):
    if len(all_rows) == 0:
        return None

    for i in range(len(sql_cols)):
        if sql_cols[i] == "address":
            sql_cols[i] = "poi_address"

    static_keys = sql_cols[:1]
    loop_keys = sql_cols[1:]

    if len(loop_keys) == 0:
        assert len(all_rows[0]) - len(static_keys) == 0
        loop = 0
    else:
        assert (len(all_rows[0]) - len(static_keys)) % len(loop_keys) == 0 
        loop = int((len(all_rows[0]) - len(static_keys)) / len(loop_keys))

    resolved_dicts = []
    
    for row in all_rows:
        resolved_dict = base_resolved_dict.copy()
        resolved_dict["@poi_type_1"] = row[0].replace(" ", "_")

        for idx1 in range(loop):
            for idx2, key in enumerate(loop_keys):
                if "distance" in key:
                    resolved_dict[f"@{key}_1{idx1+1}"] = str(row[len(static_keys)+len(loop_keys)*idx1+idx2])+"_miles"
                elif "traffic_info" in key:
                    resolved_dict[f"@{key}_1{idx1+1}"] = Num2Traffic[row[len(static_keys)+len(loop_keys)*idx1+idx2]]
                else:
                    resolved_dict[f"@{key}_1{idx1+1}"] = row[len(static_keys)+len(loop_keys)*idx1+idx2].replace(" ", "_")
        resolved_dicts.append(resolved_dict.copy())  
    return resolved_dicts

def get_schedule_entities(all_rows, sql_cols, base_resolved_dict):
    if len(all_rows) == 0:
        return None

    static_keys = sql_cols[:1]
    loop_keys = sql_cols[1:]

    if len(loop_keys) == 0:
        assert len(all_rows[0]) - len(static_keys) == 0
        loop = 0
    else:
        assert (len(all_rows[0]) - len(static_keys)) % len(loop_keys) == 0 
        loop = int((len(all_rows[0]) - len(static_keys)) / len(loop_keys))

    resolved_dicts = []
    
    for row in all_rows:
        resolved_dict = base_resolved_dict.copy()
        resolved_dict[f"@event_1"] = row[0].replace(" ", "_")

        for idx1 in range(loop):
            for idx2, key in enumerate(loop_keys):
                resolved_dict[f"@{key}_1{idx1+1}"] = row[len(static_keys)+len(loop_keys)*idx1+idx2].replace(" ", "_")
        resolved_dicts.append(resolved_dict.copy())  
    return resolved_dicts

def get_entities(domain, base_resolved_dict, kb_path, sql, sql_cols, constraints):
    conn = sqlite3.connect(kb_path)
    c = conn.cursor()
    c.execute(sql)
    all_rows = c.fetchall()

    if domain == "weather":
        return get_weather_entities(all_rows, sql_cols, base_resolved_dict)
    elif domain == "schedule":
        return get_schedule_entities(all_rows, sql_cols, base_resolved_dict)
    elif domain == "navigate":
        return get_navigate_entities(all_rows, sql_cols, base_resolved_dict)
    else:
        raise ValueError("Invalid domain!")

def fill_constraints(constraints):
    resolved_dict = {}
    # Possible constraints: 
    # @today  0 -> text:today, meaning:monday
    if "@today" in constraints:
        if "0" in constraints["@today"]:
            resolved_dict["@today_00"] = "today"
        if "1" in constraints["@today"]:
            resolved_dict["@today_01"] = "current"
        if "2" in constraints["@today"]:
            resolved_dict["@today_02"] = "currently"
    # @weekly_time: could make a list 0: week 1: today 2: tmr
    if "@weekly_time" in constraints:
        if "0" in constraints["@weekly_time"]:
            resolved_dict["@weekly_time_00"] = "week"
        if "1" in constraints["@weekly_time"]:
            resolved_dict["@weekly_time_01"] = "today"
        if "2" in constraints["@weekly_time"]:
            resolved_dict["@weekly_time_02"] = "tomorrow"
        if "3" in constraints["@weekly_time"]:
            resolved_dict["@weekly_time_03"] = "weekend"
        if "6" in constraints["@weekly_time"]:
            resolved_dict["@weekly_time_06"] = random.sample(["next_week", "this week", "next_few_days", "next 7 days"], 1)[0]
    # navigate domain
    for i in range(1,9):
        if f"@traffic_info_{i}" in constraints:
            if -1 in constraints[f"@traffic_info_{i}"]:
                resolved_dict[f"@traffic_info_{i}0"] = "heavy_traffic"
            elif -2 in constraints[f"@traffic_info_{i}"]:
                resolved_dict[f"@traffic_info_{i}0"] = "no_traffic"
    return resolved_dict

def generate_weather_dialogues(kb_path, dialogue, delex_to_chat_dict, delex_resolved_args_list, key_group, constraints, rain_record):
    resolved_dicts = []
    str_dialogues = []
    
    # consider contraints in key "0"
    negation = False
    # assume there's no multiple constraints for each key
    for key in key_group["1"]:
        i = 0
        ent = key_group["1"][key]
        if "0" in ent:
            # detected constraint exists
            if key == "@weather_attribute": # negation
                negation = True
                constraints["negation"] = 0 # weekly negation
            if key == "@temperature_high" or key == "@temperature_low":
                constraints[f"{key}_{i+1}"] = [] 
                constraints[f"{key}_{i+1}"].append(0) 
                if ("@weekly_time" in constraints and ("0" in constraints["@weekly_time"] or "6" in constraints["@weekly_time"])) or "@weekly_time" not in constraints:
                    constraints["max_min_week"] = True
                else:
                    constraints["max_min"] = True
            
    # consider constraints that are not detected in delex process
    # go through ```dialogue``` to match key words
    # TODO enable weekly_negation
    # for turn_id, request, response in dialogue:
    #     for sent in [request.str, response.str]:                         
    #         if negation and "@date" in sent and "@weather_attribute_10" in sent:
    #             constraints["negation"] = 1 # not weekly negation

    base_resolved_dict = fill_constraints(constraints)
    
    # only handle "1" group TODO extend to "2"
    sql, sql_cols = generate_sql("weather", key_group["1"], constraints) # from key group and constraints generate sql
    resolved_dicts = get_entities("weather", base_resolved_dict, kb_path, sql, sql_cols, constraints) # use sql to query the KB and get the possible entities

    for resolved_dict in resolved_dicts:
        # Generate new dialogue
        for delex_word, knowledge_value in resolved_dict.items():
            if delex_word in delex_to_chat_dict:
                for chat in delex_to_chat_dict[delex_word]:
                    if knowledge_value == "rain" and rain_record is not None:
                        location_delex = delex_word[:-1].replace("weather_attribute", "location")
                        location_value = resolved_dict[location_delex]
                        for item in rain_record:
                            if item[0] == location_value:
                                knowledge_value = "raining"
                    else:
                        knowledge_value = knowledge_value
                    chat.str = chat.str.replace(delex_word, knowledge_value)
        
        # Detect "word y" "word ing"
        # Generate string version of the new dialogue
        str_dialogue = []
        for turn_id, request, response in dialogue:
            str_dialogue.append((turn_id, request.str.replace(" y ", "").replace(" ing ", ""), response.str.replace(" y ", "").replace(" ing ", "")))

        # Reset all RevertibleString to the original chat
        for delex_word in delex_to_chat_dict.keys():
            for chat in delex_to_chat_dict[delex_word]:
                chat.to_origin()

        str_dialogues.append(str_dialogue)
    assert len(str_dialogues) > 0
    return str_dialogues

def generate_schedule_dialogues(kb_path, dialogue, delex_to_chat_dict, delex_resolved_args_list, key_group, constraints):
    resolved_dicts = []
    str_dialogues = []

    base_resolved_dict = fill_constraints(constraints)
    # from key group and constraints generate sql
    # use sql to query the KB and get the possible entities
    # only handle "1" group TODO extend to "2"
    sql, sql_cols = generate_sql("schedule", key_group["1"], constraints)
    resolved_dicts = get_entities("schedule", base_resolved_dict, kb_path, sql, sql_cols, constraints)
    if resolved_dicts is None:
        return None

    for resolved_dict in resolved_dicts:
        # Generate new dialogue
        for delex_word, knowledge_value in resolved_dict.items():
            if delex_word in delex_to_chat_dict:
                for chat in delex_to_chat_dict[delex_word]:
                    chat.str = chat.str.replace(delex_word, knowledge_value.lower())
        
        # Detect "word y" "word ing"
        # Generate string version of the new dialogue
        str_dialogue = []
        for turn_id, request, response in dialogue:
            str_dialogue.append((turn_id, request.str, response.str))

        # Reset all RevertibleString to the original chat
        for delex_word in delex_to_chat_dict.keys():
            for chat in delex_to_chat_dict[delex_word]:
                chat.to_origin()

        str_dialogues.append(str_dialogue)
    assert len(str_dialogues) > 0
    return str_dialogues

def generate_navigate_dialogues(kb_path, dialogue, delex_to_chat_dict, delex_resolved_args_list, key_group, constraints):
    resolved_dicts = []
    str_dialogues = []
    
    # consider contraints in key "0"
    # assume there's no multiple constraints for each key
    for key in key_group["1"]:
        i = 0
        ent = key_group["1"][key]
        if "0" in ent:
            # detected constraint exists
            if key == "@traffic_info": # may be "heavy_traffic" / "no_traffic"
                if "avoid" in " ".join([ sent.str for sent in delex_to_chat_dict[f"{key}_{i+1}{0}"]]):
                    constraints[f"{key}_{i+1}"] = [] 
                    constraints[f"{key}_{i+1}"].append(-1) # negation: avoid heavy_traffic
                else:
                    constraints[f"{key}_{i+1}"] = [] 
                    constraints[f"{key}_{i+1}"].append(-2) # no_traffic, soft constraint, ignore
            
    # consider constraints that are not detected in delex process
    for turn_id, request, response in dialogue: 
        for sent in [request.str, response.str]: # go through ```dialogue``` to match key words                        
            # navigate domain: 
            if ("less traffic" in sent) or ("least traffic" in sent): # 1. traffic_info: "less traffic" \ "least traffic" \ "avoid @traffic_info_x0"(heavy_traffic) -1
                if f"@traffic_info_{i+1}" not in constraints:
                    constraints[f"@traffic_info_{i+1}"] = [] 
                constraints[f"@traffic_info_{i+1}"].append(0) # choose the one with less traffic
                constraints["max_min"] = True
            
            if ("closest" in sent) or ("quickest" in sent) or ("nearest" in sent) or ("shortest" in sent): # 2. distance: "closest" \ "quickest" \
                if f"@distance_{i+1}" not in constraints:
                    constraints[f"@distance_{i+1}"] = [] 
                constraints[f"@distance_{i+1}"].append(0) # choose the one with less distance 
                constraints["max_min"] = True 

    base_resolved_dict = fill_constraints(constraints)
    # from key group and constraints generate sql
    # use sql to query the KB and get the possible entities
    # only handle "1" group TODO extend to "2"
    sql, sql_cols = generate_sql("navigate", key_group["1"], constraints)
    resolved_dicts = get_entities("navigate", base_resolved_dict, kb_path, sql, sql_cols, constraints)
    if resolved_dicts is None:
        return None

    for resolved_dict in resolved_dicts:
        # Generate new dialogue
        for delex_word, knowledge_value in resolved_dict.items():
            if delex_word in delex_to_chat_dict:
                for chat in delex_to_chat_dict[delex_word]:
                    chat.str = chat.str.replace(delex_word, knowledge_value.lower())
        
        # Detect "word y" "word ing"
        # Generate string version of the new dialogue
        str_dialogue = []
        for turn_id, request, response in dialogue:
            str_dialogue.append((turn_id, request.str, response.str))

        # Reset all RevertibleString to the original chat
        for delex_word in delex_to_chat_dict.keys():
            for chat in delex_to_chat_dict[delex_word]:
                chat.to_origin()

        str_dialogues.append(str_dialogue)
    assert len(str_dialogues) > 0
    return str_dialogues

def generate_dialogues(kb_path, meta_dialogue, rain_record):
    dialogue, delex_to_chat_dict, delex_resolved_args_list, dialog_type = meta_dialogue

    poi = Type2POI["schedule"]

    # get maximum entity values for entities
    key_group = {}
    for meta in delex_resolved_args_list:
        last_underscore_index = meta.rindex('_')
        delex_key = meta[:last_underscore_index]
        delex_index = meta[last_underscore_index+1]
        if (delex_key == poi):
            dist_index = ""
        else:
            dist_index = meta[-1]

        if delex_key != poi:
            if delex_index not in key_group:
                key_group[delex_index] = {}
            if delex_key not in key_group[delex_index]:
                key_group[delex_index][delex_key] = []
            key_group[delex_index][delex_key].append(dist_index)
    
    if dialog_type == "schedule":
        for i in key_group:
            key_dict = key_group[i]
            for turn_id, request, response in dialogue:
                for sent in [request.str, response.str]:
                    expand_num = max([len(item) for item in list(key_dict.values())])
                    for key in key_dict:
                        if len(key_dict[key]) < expand_num:
                            expand_key = key_dict[key] * int(expand_num / len(key_dict[key]))
                            key_dict[key] = expand_key
    
    if "0" in key_group:
        constraints = key_group["0"]
    else:
        constraints = {}
    
    if dialog_type == "weather":
        str_dialogues = generate_weather_dialogues(kb_path, dialogue, delex_to_chat_dict, delex_resolved_args_list, key_group, constraints, rain_record)
    elif dialog_type == "schedule":
        str_dialogues = generate_schedule_dialogues(kb_path, dialogue, delex_to_chat_dict, delex_resolved_args_list, key_group, constraints)
    else:
        str_dialogues = generate_navigate_dialogues(kb_path, dialogue, delex_to_chat_dict, delex_resolved_args_list, key_group, constraints)
    return str_dialogues


def generte_dialog_test_set(d, knowledge_folder, dialogue_path, num_augmented_dialogue, kb_types, split, output_folder, distribution):
    num_sample = len(kb_types[split])
    for i in tqdm(range(num_sample), total=num_sample, ncols=100):
        domain = kb_types[split][i]
        if domain != d:
            continue
        # Read KB & dataset
        # print("Read KB & dataset ... ")
        kb_path = os.path.join(knowledge_folder+"/"+split, f"dialog_{i}.db") 

        if not os.path.exists(kb_path):
            with open(os.path.join(output_folder, f"dialog_{i}.txt"), "w") as f:
                f.write("")
            continue

        dialogues = read_dialogue(dialogue_path)

        # Generate dialog index
        index_dialog = generate_dialogue_index(dialogues)

        with open(f"{knowledge_folder}/{split}_rain_record.json", "r") as f:
            rain_record_map = json.load(f)
        
        if str(i) in rain_record_map:
            rain_record = rain_record_map[str(i)]
        else:
            rain_record = None

        meta_dialogues = index_dialog.loc[index_dialog['domain'] == domain, 'meta'][:num_augmented_dialogue]
        str_dialogues = []
        for template_id, meta_dialogue in enumerate(meta_dialogues):
            # print("[template_id]", template_id)
            str_dialogue = generate_dialogues(kb_path, meta_dialogue, rain_record)
            if str_dialogue is not None:
                str_dialogues.extend(str_dialogue)
                if distribution:
                    if template_id not in DIST[domain]:
                        DIST[domain][template_id] = []
                    DIST[domain][template_id].append(len(str_dialogue))
            else:
                if distribution:
                    if template_id not in DIST[domain]:
                        DIST[domain][template_id] = []
                    DIST[domain][template_id].append(0)

        assert len(str_dialogues) > 0
        dump_dialogue_to_file(str_dialogues, os.path.join(output_folder, f"dialog_{i}.txt"))

    if distribution:
        with open(f"./SMD/distribution_{split}.json", "w") as f:
            json.dump(DIST, f)

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(description='Generation SMD')
    parser.add_argument('--dialogue_path', type=str, default='./SMD/weather_template.txt', help='dialog path, default: ./SMD/dialog_manual_template.txt')
    parser.add_argument('--knowledge_folder', type=str, default='./SMD/KBs-test', help='knowledge base folder, default: ./SMD/KBs')
    parser.add_argument('--output_folder', type=str, default='./SMD', help='output folder path for generation result, default: ./SMD')
    parser.add_argument('--domain', type=str, default="navigate", help='dialogue domain and KB domain, default: schedule')
    parser.add_argument('--num_augmented_knowledge', type=int, default=10, help='number of augmented knowledge, default: 10')
    parser.add_argument('--num_augmented_dialogue', type=int, default=10, help='number of augmented dialogue, default: 10')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed for reproducible sampling, default: 0')
    parser.add_argument('--split', type=str, default="train", help='KB source, default: train')
    parser.add_argument('--distribute', action="store_true", help='whether to do generated data statistic')
    parser.add_argument('--build_db', action="store_true", help='whether to do build database from the dataset')
    args = vars(parser.parse_args())

    # Print begin information
    print('== Selective Generation SMD Dialogue ==')
    print(args)

    # Build DB if needed
    if args["build_db"]:
        build_db(args["knowledge_folder"], args["split"])
    
    exit()
    
    # Start Timer
    start = timeit.default_timer()  
    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth',300)
    
    # Args
    dialogue_path = args["dialogue_path"]
    knowledge_folder = args["knowledge_folder"]
    domain = args['domain']
    split = args['split']
    num_augmented_knowledge = args['num_augmented_knowledge']
    num_augmented_dialogue = args['num_augmented_dialogue']
    num_augmented_dialogue = min(num_augmented_dialogue, Type2NumKB[split][domain])

    random_seed = args['random_seed']
    output_folder = args['output_folder']
    output_path = f'{output_folder}/gen-smd-{domain}-nk{num_augmented_knowledge}-nd{num_augmented_dialogue}.txt'
    random.seed (random_seed)
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    if split in ["dev", "test"]:
        # load kb_types json file
        with open(os.path.join(knowledge_folder, "kb_types.json"), "r") as f:
            kb_types = json.load(f)
        generte_dialog_test_set(domain, knowledge_folder, dialogue_path, num_augmented_dialogue, kb_types, split, output_folder, args["distribute"])
        
        # Print Execution time
        stop = timeit.default_timer()

        # Print end information
        print(f'File saved on `{output_folder}`')
        print(f'Execution time: {stop - start:.4}s') 
        exit()
    
