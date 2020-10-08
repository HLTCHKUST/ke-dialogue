## import csv
import json
import itertools
import pprint
import random
from collections import Counter
import itertools

import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from neo4j import GraphDatabase
from neo4j import unit_of_work
from tqdm import tqdm
import pickle

from revertible_string import RevertibleString

# DialogKG metadata

# DialogMeta
#    entity to chat
#    chat dialogue
#    dialog entities
#    subgraph meta

# SubgraphMeta
#    variable to entity map
#    entity to variable map
#    template subgraph
#    subgraph start node

###
# neo4j function
###
def drop_count_index(tx, suffix=None):
    feature_name = 'count'
    if suffix:
        feature_name = f'count_{suffix}'
    tx.run(f"DROP INDEX ON :Node({feature_name})")

def create_count_index(tx, suffix=None):
    feature_name = 'count'
    if suffix:
        feature_name = f'count_{suffix}'
    tx.run(f"CREATE INDEX ON :Node({feature_name})")
    
def warmup_cache(tx):
    tx.run("CALL apoc.warmup.run()")

def reset_count(tx, count=None, suffix=None):
    feature_name = 'count'
    if suffix:
        feature_name = f'count_{suffix}'
        
    if count:
        tx.run(f"MATCH (a:Node) SET a.{feature_name} = {count}")
    else:
        tx.run(f"MATCH (a:Node) SET a.{feature_name}  = SIZE((a)-[]-())")

def decrement_count(tx, node_values, suffix=None):
    feature_name = 'count'
    if suffix:
        feature_name = f'count_{suffix}'
    tx.run(f"MATCH (a:Node) WHERE a.value IN {str(node_values)} SET a.{feature_name} = a.{feature_name} - 1")
    
def run_query(tx, query):
    result_list = []
    for record in tx.run(query):
        result_list.append(record)
    
    # Return result as list
    return result_list

###
# Reader & Writer
##
def execute_warmup_cache(neo4j_driver):
    with neo4j_driver.session() as session:
        return session.write_transaction(warmup_cache)
    
def execute_reset_count(neo4j_driver, count=None, suffix=None):
    with neo4j_driver.session() as session:
        session.write_transaction(reset_count, count, suffix)
    
def execute_decrement_count(neo4j_driver, node_values, suffix=None):
    with neo4j_driver.session() as session:
        session.write_transaction(decrement_count, node_values, suffix)
        
def execute_write_fn(neo4j_driver, query_fn):
    with neo4j_driver.session() as session:
        return session.write_transaction(query_fn)
    
@unit_of_work(timeout=30.0)
def execute_query(neo4j_driver, query):
    with neo4j_driver.session() as session:
        return session.read_transaction(run_query, query)

@unit_of_work(timeout=30.0)
def execute_retrieve_first(neo4j_driver, query):
    with neo4j_driver.session() as session:
        for record in session.run(query):
            return record

def execute_retrieve_node_count(neo4j_driver, suffix=None):
    feature_name = 'count'
    if suffix:
        feature_name = f'count_{suffix}'
        
    with neo4j_driver.session() as session:
        result_dict = {}
        for record in session.run(f'MATCH (n:Node) RETURN n.value AS name, n.{feature_name} AS count'):
            result_dict[record['name']] = record['count']
        return result_dict
    
###
# Common function
###
def get_global_entity_DIALKG():
    with open('./opendialkg/opendialkg_entities.txt') as f:
        global_entity_list = []
        for x in f:
            global_entity_list.append(x.replace("\n",""))
    return list(set(global_entity_list))

def substring_sieve(string_list):
    string_list.sort(key=lambda s: len(s), reverse=True)
    out = []
    for s in string_list:
        if not any([s in o for o in out]):
            out.append(s)
    return out

###
# Subgraph metadata generation function
###
def retrieve_spanning_tree(tx, query):
    """
        Input: 
            tx             handle for querying neo4j
            query          APOC query to be executed
            
        Sample query for builiding spanning tree
        MATCH
            (n0:Node{value:'The Graveyard Book'}), 
            (n1:Node{value:'Wizard and Glass'}), 
            (n2:Node{value:'Dave McKean'}),
        CALL apoc.path.spanningTree(n0,{maxLevel:5, endNodes:[n1,n2]})
        YIELD path
        RETURN path;
    """
    result_set = set() # Result buffer
    
    # Collect result
    for record in tx.run(query):
        for path in record:
            for rel in path:
                result_set.add((rel.start_node['value'], rel.type, rel.end_node['value']))
    
    # Return result as list
    return list(result_set)

def generate_subgraph(neo4j_driver, node_values):
    """
        Input: 
            neo4j_driver        neo4j driver
            node_values         node values for generating the spanning tree subgraph
        Output:
            subgraph            a subgraph represented as list of unique (s,r,o) tuple
    """
    # Preprocessing
    node_values = [node_value.replace("'","\\'") for node_value in node_values]
    
    # Build neo4j cypher query
    match_nodes = ', '.join([f"(n{i}:Node{{value:'{value}'}})" for i,value in enumerate(node_values)])
    max_level = len(node_values) + 3

    # Random sample source & end nodes
    nodes = [f'n{i}'for i in range(len(node_values))]
    best_subgraph = None
    best_start_node = None
    for i in range(len(node_values)):
        start_node = nodes[i]
        end_nodes = ', '.join(nodes[:i] + nodes[i+1:])

        spanning_tree_query =  f"""
            MATCH {match_nodes}
            CALL apoc.path.spanningTree({start_node},{{maxLevel:{max_level}, endNodes:[{end_nodes}]}})
            YIELD path
            RETURN path;
        """
        

        with neo4j_driver.session() as session:
            subgraph = session.read_transaction(retrieve_spanning_tree, spanning_tree_query)
            
        # Pick the best spanning tree
        if not best_subgraph:
            best_subgraph = subgraph
        elif len(subgraph) > 0 and len(subgraph) < len(best_subgraph):
            best_subgraph = subgraph
    
    entity_counter = Counter()
    for s,r,o in best_subgraph:
        entity_counter[s] += 1
        entity_counter[o] += 1
    best_start_node = start_node
            
    # Return best spanning three
    return best_subgraph, best_start_node
    
def extract_subgraph_meta(subgraph, start_node):
    """
        Input: 
            subgraph                 a subgraph represented as list of unique (s,r,o) tuple
        Output:
            entity_to_var            map from entity to variable
            var_to_entity            map from variable to entity
            template_subgraph        a template subgraph represented as list of unique (var_s,r,var_o) tuple
            subgraph_start_node      string variable name of the starting node
    """    
    # Sort subgraph by relationship, so we can match two equivalent subgraphs later
    subgraph.sort(key=lambda sro: sro[1])
    
    # Retrieve entity
    entity_set = set()
    for s, r, o in subgraph:
        entity_set.add(s)
        entity_set.add(o)
    
    # Generate map entity to variable & variable to entity
    entities = list(entity_set)
    entities.sort()
    
    entity_to_var = {e:i for i,e in enumerate(entities)}
    var_to_entity = {i:e for i,e in enumerate(entities)}
    
    # Construct template subgraph
    template_subgraph = []
    for s, r, o in subgraph:
        template_subgraph.append((entity_to_var[s], r, entity_to_var[o]))
    
    # Return result as list
    return entity_to_var, var_to_entity, template_subgraph, start_node

###
# Dialog extraction function
###
def get_match_entities(text,entities, cased=False):
    match_entities = []
    for entity in entities:
        if not cased:
            is_entity_exist = entity.lower() in text.lower()
        else:
            is_entity_exist = entity in text
            
        if(is_entity_exist):
            match_entities.append(entity)
    match_entities = substring_sieve(match_entities)
    return match_entities

def extract_dialog(dialogue, global_ent, cased=False):
    """
        Input:
            dialogue                dialogue conversation            list[{speaker:<str>, text:<str>}]
            global_ent              list of all possible entities    list[<str>]
        Output:
            DialogKG Metadata
               - chat_dialogue      chat dialog with RevertibleString text       list[{speaker:<str>, text:<RevertibleString>}]
               - entity_to_chat     map from entity to RevertibleString text     dict{<str>:list[<RevertibleString>]}
               - dialog_entities    list of all entities in the dialog           list[<str>]
    """  
    chat_dialogue = []
    entity_to_chat = {}    
    dialogue_entities = []
    
    for i, conv in enumerate(dialogue):
        match_entities = get_match_entities(conv["text"].replace("  "," "), global_ent, cased)
        dialogue_entities += match_entities

        text = RevertibleString(conv["text"].replace("  "," "))
        chat_dialogue.append({"speaker":conv['speaker'], "text":text})
        
        for entity in match_entities:
            if entity not in entity_to_chat:
                entity_to_chat[entity] = []
            entity_to_chat[entity].append(text)

    # Clean the entity matched
    # Remove substring included in large string ==> e.g. [Aircraft,Air] --> [Aircraft]
    dialogue_entities = substring_sieve(dialogue_entities)
    dialogue_entities = list(filter(lambda x: x.isnumeric() or len(x) >= 5, dialogue_entities))
    dialogue_entities = list(filter(lambda x: x.lower()!= 'author', dialogue_entities))
    dialogue_entities = list(filter(lambda x: x.lower()!= 'no problem', dialogue_entities))
    dialogue_entities = list(filter(lambda x: x.lower()!= 'you again', dialogue_entities))
    dialogue_entities = list(filter(lambda x: x.lower()!= 'watch', dialogue_entities))
    dialogue_entities = list(filter(lambda x: x.lower()!= 'information', dialogue_entities))
    dialogue_entities = list(filter(lambda x: x.lower()!= 'haven', dialogue_entities))
    dialogue_entities = list(filter(lambda x: x.lower()!= 'review', dialogue_entities))
    dialogue_entities = list(filter(lambda x: x.lower()!= 'check it out', dialogue_entities))
    dialogue_entities = list(filter(lambda x: x.lower()!= 'for you', dialogue_entities))
    
    # Remove filtered entities from entity_to_chat
    deleted_entities = []
    for key in entity_to_chat.keys():
        if key not in dialogue_entities:
            deleted_entities.append(key)
    
    for entity in deleted_entities:
         del entity_to_chat[entity]
    
    # Return dialog meta
    return chat_dialogue, entity_to_chat, dialogue_entities

###
# Meta generation function
###
def generate_dialog_meta(dialogue, global_ent, neo4j_driver, cased=False):
    """
        Input:
            dialogue            dialogue conversation            list[{speaker:<str>, text:<str>}]
            global_ent          list of all possible entities    list[<str>]
            neo4j_driver        neo4j driver
        Output:
            DialogKG Metadata
               chat_dialogue                   list[{speaker:<str>, text:<RevertibleString>}]
               entity_to_chat                  dict{<str>:list[<RevertibleString>]}
               dialog_entities                 list[<str>]
               subgraph_meta                   (dict, dict, list)
                   variable_to_entity_map           dict{var:<str>}
                   entity_to_variable_map           dict{<str>:var}
                   template_subgraph                list<(s,r,o)>
                   subgraph_start_node              str
    """  
    chat_dialogue, entity_to_chat, dialogue_entities = extract_dialog(dialogue, global_ent, cased)
    if len(dialogue_entities) > 0:
        subgraph, start_node = generate_subgraph(neo4j_driver, dialogue_entities)
        subgraph_meta = extract_subgraph_meta(subgraph, start_node)
    else:
        subgraph_meta = (dict(), dict(), list(), None)
    return chat_dialogue, entity_to_chat, dialogue_entities, subgraph_meta

def generate_dataset_meta(data_path, global_ent, neo4j_driver, cased=False):
    """
        Input:
            data_path           path of dialogKG dataset         <str>
            global_ent          list of all possible entities    list[<str>]
            neo4j_driver        neo4j driver
        Output:
            DialogKG Metadata
               chat_dialogue                   list[{speaker:<str>, text:<RevertibleString>}]
               entity_to_chat                  dict{<str>:list[<RevertibleString>]}
               dialog_entities                 list[<str>]
               subgraph_meta                   (dict, dict, list)
                   variable_to_entity_map           dict{var:<str>}
                   entity_to_variable_map           dict{<str>:var}
                   template_subgraph                list<(s,r,o)>
                   subgraph_start_node              str
    """  
    df = pd.read_csv(data_path)
    df.columns = ['message','user_rating','assistant_rating']
    
    dialogue_metas = []
    for row in df.itertuples():
        conversation = []
        KB = []
        for t in eval(row.message):
            if('message' in t):
                conversation.append({"speaker":t['sender'], "text":t['message'], "gold-kb":[]})
            elif "path" in t['metadata']:
                KB += t['metadata']['path'][1]
        dialogue_meta = generate_dialog_meta(conversation,global_ent, neo4j_driver, cased)
        dialogue_metas.append(dialogue_meta)
    return dialogue_metas

###
# Dialog generation and saving functions
###
def generate_dialogue(neo4j_driver, dialogue_meta, limit=1, is_parallel=False, return_entity=False, suffix=None):
    chat_dialogue, entity_to_chat, dialogue_entities, subgraph_meta = dialogue_meta
    entity_to_var, var_to_entity, template_subgraph, subgraph_start_node = subgraph_meta

    # Set count feature with suffix
    feature_name = 'count'
    if suffix:
        feature_name = f'count_{suffix}'

    # Warmup cache
    execute_warmup_cache(neo4j_driver)
    
    # generate Neo4J query using the extracted graph pattern
    match_nodes = []
    if is_parallel:
        f_node = subgraph_start_node # first node for parallelization
        f_rel = [] # first node relations for parallelization
        query_nodes = set()
        for s,r,o in template_subgraph:
            if f'n{s}' == subgraph_start_node:
                f_rel.append(r)
                match_nodes.append(f"(_)-[:{r}]->(n{o})")
                query_nodes.update(['_',f'n{o}'])
            elif f'n{o}' == subgraph_start_node:
                f_rel.append(r)
                match_nodes.append(f"(n{s})-[:{r}]->(_)")
                query_nodes.update([f'n{s}','_'])
            else:
                match_nodes.append(f"(n{s})-[:{r}]->(n{o})")
                query_nodes.update([f'n{s}',f'n{o}'])

        str_match_nodes = ', '.join(match_nodes)
        str_query_nodes = ', '.join(list(query_nodes))
        str_query_where = ' AND '.join([f'{node}.{feature_name} > 0'for node in query_nodes])
        str_outer_match_nodes = ', '.join([f'(n:Node)-[:{rel}]-(:Node)' for rel in f_rel])
        query = f"""
            MATCH {str_outer_match_nodes} WHERE n.{feature_name} > 0
            WITH COLLECT(DISTINCT n) AS f_node
            CALL apoc.cypher.mapParallel("
                MATCH {str_match_nodes} 
                WHERE {str_query_where} 
                RETURN {str_query_nodes}
                LIMIT {limit}
            ",{{parallel:true}}, f_node) YIELD value
            RETURN value
        """
        
        query_result = execute_retrieve_first(neo4j_driver, query)
        if query_result:
            query_result = query_result['value']
    else:
        query_nodes = set()
        for s,r,o in template_subgraph:
            match_nodes.append(f"(n{s})-[:{r}]->(n{o})")
            query_nodes.update([f'n{s}',f'n{o}'])

        str_match_nodes = ', '.join(match_nodes)
        str_query_nodes = ', '.join(list(query_nodes))
        str_query_where = ' AND '.join([f'{node}.{feature_name} > 0'for node in query_nodes])
        str_query_order = ' + '.join([f'{node}.{feature_name}' for node in query_nodes])
        
        query = f"""
            MATCH {str_match_nodes} 
            WHERE {str_query_where} 
            RETURN {str_query_nodes}
            LIMIT {limit}
        """

        query_result = execute_retrieve_first(neo4j_driver, query)
    
    # Handling for no result
    if not query_result:
        return None

    str_dialogues = []
    var_to_ent_result = {}
    used_entities = [] # Buffer for decrement count
    for key, node in query_result.items():
        var, new_entity = int(key[1:] if key != '_' else f_node[1:]), node['value']
        src_entity = var_to_entity[var]
        if src_entity in entity_to_chat:
            # Modify chat
            chats = entity_to_chat[src_entity]
            for chat in chats:
                # Lower case entity
                lower_src_entity = src_entity.lower()
                entity_index = chat.str.lower().index(lower_src_entity)
                chat.str = chat.str[:entity_index] + lower_src_entity + chat.str[entity_index + len(lower_src_entity):]

                # Replace entity
                chat.str = chat.str.replace(lower_src_entity, new_entity)
                chat.entities.append(new_entity)

            # Add new entity to used list
            used_entities.append(new_entity)

    # Generate string version of the new dialogue
    str_dialogue = []
    for conv in chat_dialogue:
        str_dialogue.append({
            'speaker':conv['speaker'], 'text':conv['text'].str, 'entities':conv['text'].entities, "gold_KB":[]
        })

    # Reset all RevertibleString to the original chat
    for entity in entity_to_chat.keys():
        for chat in entity_to_chat[entity]:
            chat.to_origin()

    # Decrement all used entity
    execute_decrement_count(neo4j_driver, used_entities, suffix=suffix)

    # Return the generated dialog
    if return_entity: 
        return str_dialogue, used_entities
    return str_dialogue

def write_dialogue_to_file(generated_dialogues, batch_size, random_seed):
    json.dump(generated_dialogues, open(f"./opendialkg/generated_dialogue_bs{batch_size}_rs{random_seed}.json","w"))
