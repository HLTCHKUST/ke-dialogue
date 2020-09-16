import json
import os.path
from utils.eval_metrics import moses_multi_bleu
import glob as glob
import numpy as np
import pandas as pd
import jsonlines
from tabulate import tabulate
from tqdm import tqdm
import argparse
from bert_score import score

def hasNoNumbers(inputString):
    return not any(char.isdigit() for char in inputString)

def checker_global_ent(e,gold):
    nonumber = hasNoNumbers(e)
    sub_string = True
    for g in gold:
        if(e.lower() in g.lower()):
            sub_string = False
    return sub_string and nonumber

def substringSieve(string_list):
    string_list.sort(key=lambda s: len(s), reverse=True)
    out = []
    for s in string_list:
        if not any([s.lower() in o.lower() for o in out]):
            out.append(s)
    return out

def get_match_ent(text,entity):
    matches = []
    for n in entity:
        if(n.lower() in text.lower()):
            matches.append(n.lower())
        if(n in text):
            matches.append(n.lower())
    matches = list(set(matches))
    matches = substringSieve(matches)
    matches = list(filter(lambda x: x.isnumeric() or len(x) >= 5, matches))
    matches = list(filter(lambda x: x.lower()!= 'author', matches))
    matches = list(filter(lambda x: x.lower()!= 'no problem', matches))
    matches = list(filter(lambda x: x.lower()!= 'you again', matches))
    matches = list(filter(lambda x: x.lower()!= 'watch', matches))
    matches = list(filter(lambda x: x.lower()!= 'information', matches))
    matches = list(filter(lambda x: x.lower()!= 'haven', matches))
    matches = list(filter(lambda x: x.lower()!= 'review', matches))

    return matches

def compute_prf(gold, pred, global_entity_list):#, kb_plain=None):
    # local_kb_word = [k[0] for k in kb_plain]
    TP, FP, FN = 0, 0, 0
    if len(gold)!= 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1        
        list_FP = []
        for e in get_match_ent(pred,global_entity_list):
            if(e not in gold):
                list_FP.append(e)
        FP = len(list(set(substringSieve(list_FP))))
        precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
        recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
    else:
        precision, recall, F1, count = 0, 0, 0, 0
    return F1, count

def compute_prf_prec(gold, pred, global_entity_list):#, kb_plain=None):
    # local_kb_word = [k[0] for k in kb_plain]
    TP, FP = 0, 0
    if len(gold)!= 0:
        count = 1
        list_TP = []
        list_FP = []
        for e in get_match_ent(pred,global_entity_list):
            if(e in gold):
                list_TP.append(e)
            else:
                list_FP.append(e)
        FP = len(list(set(substringSieve(list_FP))))
        TP = len(list(set(substringSieve(list_TP))))
        precision = float(TP) / float(TP+FP) if (TP+FP)!=0 else 0
    else:
        precision, count = 0, 0
    return precision, count

def get_global_entity_DIALKG():
    with open('data/opendialkg/opendialkg_entities.txt') as f:
        global_entity_list = []
        for x in f:
            global_entity_list.append(x.replace("\n",""))
    return list(set(global_entity_list))

def get_unique_entities(data_split, eval_from_graph=False):
    entities = set()
    for dialogue in tqdm(data_split):
        for d in dialogue:
            if eval_from_graph:
                if 'gold_entities' in d:
                    for e in d['gold_entities']:
                        entities.add(e)
                else:
                    for e in d['entities']:
                        entities.add(e)
            else:
                for e in d['entities']:
                    entities.add(e)

    # Filter out common entities
    entities = list(filter(lambda x: x.isnumeric() or len(x) >= 5, entities))
    entities = list(filter(lambda x: x.lower()!= 'author', entities))
    entities = list(filter(lambda x: x.lower()!= 'no problem', entities))
    entities = list(filter(lambda x: x.lower()!= 'you again', entities))
    entities = list(filter(lambda x: x.lower()!= 'watch', entities))
    entities = list(filter(lambda x: x.lower()!= 'information', entities))
    entities = list(filter(lambda x: x.lower()!= 'haven', entities))
    entities = list(filter(lambda x: x.lower()!= 'review', entities))
    entities = list(filter(lambda x: x.lower()!= 'check it out', entities))
    entities = list(filter(lambda x: x.lower()!= 'for you', entities))
    
    return entities

def score_DIALKG(model,file_to_score, test_json, global_entity_list, oov_ent_test, eval_from_graph=False, eval_prec=False):
    genr_json = json.load(open(file_to_score))
    
    if eval_from_graph:
        entity_key = 'gold_entities'
    else:
        entity_key = 'entities'
        
    if eval_prec:
        metric_fn = compute_prf_prec
    else:
        metric_fn = compute_prf
        
    GOLD, GENR = [], []
    F1_score = []
    OOV_F1_score = []
    for idx_d, dial in tqdm(enumerate(test_json),total=len(test_json)):
#     for idx_d, dial in enumerate(test_json):
        idx_t = 0
        for d in dial:
            if(d['speaker']=="assistant"):
                GOLD.append(d["text"].lower())
                GENR.append(genr_json[str(idx_d)][idx_t]["text"].lower())
                if entity_key in d:
                    gold_entities = [ent.lower() for ent in d[entity_key]]
                    oov_gold_entities = []
                    for gold in gold_entities:
                        if gold in oov_ent_test:
                            oov_gold_entities.append(gold)

                    # Calculate F1
                    F1, count = metric_fn(gold_entities, GENR[-1], global_entity_list)
                    if(count==1): 
                        F1_score.append(F1)

                    # Calculate OOV F1
                    OOV_F1, OOV_count = metric_fn(oov_gold_entities, GENR[-1], global_entity_list)
                    if(OOV_count==1): 
                        OOV_F1_score.append(OOV_F1) 

                idx_t += 1
    BLEU = moses_multi_bleu(np.array(GENR),np.array(GOLD))
    BERT_P, BERT_R, BERT_F1 = score(GENR, GOLD, lang="en")

    return {"Model":model,
            "BLEU":BLEU, 
            "BERT PREC":100*BERT_P.mean().item(),
            "BERT REC":100*BERT_R.mean().item(),
            "BERT F1":100*BERT_F1.mean().item(),
            "F1":100*np.mean(F1_score), 
            "OOV F1":100*np.mean(OOV_F1_score)}

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--eval_from_graph', action="store_true", default=False)
    parser.add_argument('--eval_prec', action="store_true", default=False)
    parser.add_argument('--hop', action="store_true", default=False)
    args = vars(parser.parse_args())
    
    eval_from_graph = args['eval_from_graph']
    eval_prec = args['eval_prec']
    hop = args['hop']

    # Load Entity
    train_json = json.load(open("data/opendialkg/train.json"))
    if hop:
        test_json = json.load(open("data/opendialkg/test_prec_2.json"))
    else:
        test_json = json.load(open("data/opendialkg/test_prec.json"))
    global_entity_list = get_global_entity_DIALKG()

    # preprocess lower case
    for idx_d, dial in tqdm(enumerate(train_json),total=len(train_json)):
        for d in dial:
            if(d['speaker']=="assistant"):
                d["entities"] = [ent.lower() for ent in d["entities"]]                    
                if 'gold_entities' in d:
                    if len(d["gold_entities"]) > 0:
                        d["gold_entities"] = [ent.lower() for ent in d["gold_entities"]] + d["entities"]
                    else:
                        del d["gold_entities"]
                              
    for idx_d, dial in tqdm(enumerate(test_json),total=len(test_json)):
        for d in dial:
            if(d['speaker']=="assistant"):
                d["entities"] = [ent.lower() for ent in d["entities"]]
                if 'gold_entities' in d:
                    if len(d["gold_entities"]) > 0:
                        d["gold_entities"] = [ent.lower() for ent in d["gold_entities"]] + d["entities"]
                    else:
                        del d["gold_entities"]
                    
    ent_train = set(get_unique_entities(train_json))
    ent_test = set(get_unique_entities(test_json, eval_from_graph=eval_from_graph))
    oov_ent_test = set(ent_test - ent_train)
    
    print('len(ent_test)', len(ent_test))
    print('len(oov_ent_test)', len(oov_ent_test))

    rows = []
    for f in glob.glob("runs/*"):
        if("DIALKG" in f and os.path.isfile(f+'/result.json')):
            print(f)
            d_name,model,_,graph,_,adj,_,edge,_,unilm,_,flattenKB,_,hist_L,_,lr,_,epoch,*_ = f.split("/")[1].split("_")
            st = model
            if(eval(graph)): st+= "+NODE "
            if(eval(adj)): st+= "+ADJ "
            if(eval(edge)): st+= "+EDGES "
            if(eval(unilm)): st+= "+UNI "
            if(eval(flattenKB)): st+= "+KB "
            kbpercentage = 0
            if 'kbpercentage_' in f:
                kbpercentage = f[f.index('kbpercentage_') + len('kbpercentage_'):].split('_')[0]
                st += f' +KB {kbpercentage}'
            st += f' ep{epoch}'
            rows.append(score_DIALKG(st,f+'/result.json',test_json, global_entity_list, oov_ent_test, eval_from_graph=eval_from_graph, eval_prec=eval_prec))
    
    # rows.append(score_SMD_GLMP("GLMPNEW",'data/SMD/GLMP/result.json'))
    print(tabulate(rows,headers="keys",tablefmt='simple',floatfmt=".2f"))
    pd.DataFrame.from_records(rows).sort_values('Model').to_csv(f'scorer_DIALKG_result_eg{eval_from_graph}_ep{eval_prec}.csv', index=False)