## import csv
import sys
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
from dialogkg_utils import *
import argparse

parser = argparse.ArgumentParser(description='DIALKG dialogue generator cased')
parser.add_argument('--random_seed', type=int, required=True, help='The random seed for template sampling')
parser.add_argument('--batch_size', type=int, default=300, help='number of template for each iteration')
parser.add_argument('--max_iteration', type=int, default=1000, help='limit number of iteration')
parser.add_argument('--stop_count', type=int, default=5, help='minimum generation count to stop generation')
parser.add_argument('--connection_string', type=str, default='bolt://eez115.ece.ust.hk:7687', help='connection string of the neo4j')
args = vars(parser.parse_args())

# Load Config
limit = 1 # This one is fixed
random_seed = args['random_seed']
batch_size = args['batch_size']
max_iteration = args['max_iteration']
stop_count = args['stop_count']
connection_string = args['connection_string']

# Setup Neo4j connection
neo4j_driver = GraphDatabase.driver(connection_string, auth=("neo4j", "CAiRE2020neo4j"))

# Load the meta file
print('Loading meta files...')
train_dialogue_metas = pickle.load(open('./opendialkg/dialogkg_train_meta_no_x_cased_reduced.pt', 'rb'))
len_subgraphs = list(map(lambda x: len(x[3][2]), train_dialogue_metas))
train_meta_df = pd.DataFrame({'graph_len':len_subgraphs, 'meta':train_dialogue_metas})

# Result Buffer
used_count_records = []
db_count_records = []
generated_dialogues = []

# Reset Count
execute_reset_count(neo4j_driver, count=None, suffix=random_seed)

# Insert initial count to df_count_records and used_entities_counter
used_count_records.append({})
db_count_records.append(execute_retrieve_node_count(neo4j_driver, suffix=random_seed))

# Iterate
print('Starting generation...')
is_finish = False
count_fulfilled_list = []
count_unfulfilled_list = []
iterations = []
iteration = 0
while not is_finish:
    count_fulfilled = 0
    count_unfulfilled = 0
    used_entities_counter = Counter()
    
    train_meta_samples = train_meta_df.loc[train_meta_df['graph_len'] <= 20,:].sample(batch_size, random_state=random_seed+sum(count_fulfilled_list))
    dialogue_metas = train_meta_samples['meta'].tolist()
    for dialogue_meta in tqdm(dialogue_metas):
        try:
            gen_dialogue, used_entities = generate_dialogue(neo4j_driver, dialogue_meta, limit, return_entity=True, suffix=random_seed)
        except KeyboardInterrupt:
            is_finish=True
            break
        except:
            gen_dialogue = None
            
        if gen_dialogue:
            count_fulfilled += 1
            generated_dialogues.append(gen_dialogue)
            for used_entity in used_entities:
                used_entities_counter[used_entity] += 1
        else:
            count_unfulfilled += 1
            
    iteration += 1
    count_fulfilled_list.append(count_fulfilled)
    count_unfulfilled_list.append(count_unfulfilled)
    iterations.append(iteration)
    
    used_count_records.append(dict(used_entities_counter))
    db_count_records.append(execute_retrieve_node_count(neo4j_driver, suffix=random_seed))

    print(f'Finish iteration {iteration} | Fulfilled: {count_fulfilled} | Total Fulfilled: {sum(count_fulfilled_list)}')
    
    # Save every 10 iteration
    if iteration % 10 == 0:
        write_dialogue_to_file(generated_dialogues, batch_size, random_seed)

        df_count_df = pd.DataFrame.from_records(db_count_records).fillna(0).astype(int).reset_index()
        used_count_df = pd.DataFrame.from_records(used_count_records).fillna(0).astype(int).reset_index()
        iteration_df = pd.DataFrame({'iteration':iterations, 'generated':count_fulfilled_list, 'failed':count_unfulfilled_list})

        df_count_df.to_csv(f'./opendialkg/db_count_records_{random_seed}.csv', index=False)
        used_count_df.to_csv(f'./opendialkg/used_count_records_{random_seed}.csv', index=False)
        iteration_df.to_csv(f'./opendialkg/generation_iteration_{random_seed}.csv', index=False)

    # Check stop condition
    if count_fulfilled <= stop_count or iteration == max_iteration:
        is_finish = True
