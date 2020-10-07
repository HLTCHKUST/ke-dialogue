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
import gc

from revertible_string import RevertibleString

from dialogkg_utils import *


## Meta Generation ##

# Preparation
batch_size = 1
count = 1
limit = 1
random.seed(0)

neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "<Password>"))
global_ent = get_global_entity_DIALKG()

# Generate Train Meta
df = pd.read_csv('./opendialkg/train.csv')
df.columns = ['Messages','User Rating','Assistant Rating']

generated_dialogues = []
dialogue_metas = []
for i, row in tqdm(enumerate(df.itertuples())):
    dialogue = []
    KB = []
    for t in eval(row.message):
        if 'message' in t:
            dialogue.append({"speaker":t['sender'], "text":t['message'], "gold_KB":[]})
        elif "path" in t['metadata']:
            KB += t['metadata']['path'][1]
    dialogue_meta = generate_dialog_meta(dialogue, global_ent, neo4j_driver, cased=True)
    dialogue_metas.append(dialogue_meta)

# write_dialogue_to_file(generated_dialogues, batch_size)            
pickle.dump(dialogue_metas, open('./opendialkg/dialogkg_train_meta.pt','wb'))
