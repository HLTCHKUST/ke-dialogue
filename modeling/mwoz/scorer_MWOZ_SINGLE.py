import json
import os.path
from utils.eval_metrics import moses_multi_bleu, huggingface_bertscore
import glob as glob
import numpy as np
import jsonlines
from tabulate import tabulate
import re
from tqdm import tqdm
import sqlite3
import editdistance
import pprint
pp = pprint.PrettyPrinter(indent=4)


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
        if not any([s in o for o in out]):
            out.append(s)
    return out

def compute_prf(pred, gold, global_entity_list):
    TP, FP, FN = 0, 0, 0
    if len(gold)!= 0:
        count = 1
        for g in gold:
            if g.lower() in pred.lower():
                TP += 1
            else:
                FN += 1
        list_FP = []
        for e in list(set(global_entity_list)):
            if e.lower() in pred.lower() and checker_global_ent(e,gold):
                if(e.lower() not in gold):
                    list_FP.append(e)
        FP = len(list(set(substringSieve(list_FP))))
        precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
        recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
    else:
        precision, recall, F1, count = 0, 0, 0, 0
    return F1, count

def get_global_entity_MWOZ():
  with open('data/MultiWOZ_2.1/ontology.json') as f:
      global_entity = json.load(f)
      global_entity_list = []
      for key in global_entity.keys():
          if(key not in ["train-book-people","restaurant-book-people",
                        "hotel-semi-stars","hotel-book-stay","hotel-book-people"]): ## this because are single numeber
            global_entity_list += global_entity[key]

      global_entity_list = list(set(global_entity_list))
  return global_entity_list


def get_entity(KB, sentence):
    list_entity = []
    for e in KB:
        if e.lower() in sentence.lower():
            list_entity.append(e.lower())
    return list_entity



def get_splits(data,test_split,val_split):
    train = {}
    valid = {}
    test  = {}
    for k, v in data.items():
        if(k in test_split):
            test[k] = v
        elif(k in val_split):
            valid[k] = v
        else:
            train[k] = v
    return train, valid, test

def get_dialog_single(gold_json,split_by_single_and_domain):
    test  = {}
    single_index = sum([ idex for n,idex in split_by_single_and_domain.items() if "single" in n and n not in ["police_single","hospital_single"]],[])
    # single_index = sum([ idex for n,idex in split_by_single_and_domain.items() if "single" in n and n not in ["taxi_single","police_single","hospital_single"]],[])

    for k, v in gold_json.items():
        if(k.lower() in single_index):
            test[k] = v
    return test

def checkin(ent,li):
    for e in li: 
        if(ent in e):
            return e
        if(editdistance.eval(e, ent)<2):
            return e
    return False


with open('data/MultiWOZ_2.1/ontology.json') as f:
    global_entity = json.load(f)
    ontology = {"train":{},"attraction":{},"hotel":{},"restaurant":{}}
    for key in global_entity.keys():
        if("semi" in key):
            domain,_,slot = key.split("-")
            if(domain in ontology): 
                ontology[domain][slot] = global_entity[key]
                
def to_query(domain, dic, reqt):
    if reqt:
        q = f"SELECT {','.join(reqt)} FROM {domain} where"
    else:
        q = f"SELECT * FROM {domain} where"
    for k,v in dic.items():
        if v == "" or v == "dontcare" or v == 'not mentioned' or v == "don't care" or v == "dont care" or v == "do n't care":
            pass
        else:
            if k == 'leaveAt':
                hour, minute = v.split(":")
                v = int(hour)*60 + int(minute)
                q += f' {k}>{v} and'
            elif k == 'arriveBy':
                hour, minute = v.split(":")
                v = int(hour)*60 + int(minute)
                q += f' {k}<{v} and'
            else:
                q += f' {k}="{v}" and'


    q = q[:-3] ## this just to remove the last AND from the query 
    return q

def align_GPT2(text):
    return text.replace("."," .").replace("?"," ?").replace(","," ,").replace("!"," !").replace("'"," '").replace("  "," ")

conn = sqlite3.connect('data/MWOZ.db')
database = conn.cursor()

dialogue_mwoz = json.load(open("data/MultiWOZ_2.1/data.json"))
test_split = open("data/MultiWOZ_2.1/testListFile.txt","r").read()
val_split = open("data/MultiWOZ_2.1/valListFile.txt","r").read()
split_by_single_and_domain = json.load(open("data/dialogue_by_domain.json"))
_,_,gold_json = get_splits(dialogue_mwoz,test_split,val_split)
gold_json = get_dialog_single(gold_json,split_by_single_and_domain)

test = []
for d in ["train","hotel","attraction","restaurant","taxi"]:
# for d in ["train","hotel","attraction","restaurant"]:
    test += [v for k, v in json.load(open(f"data/MultiWOZ_2.1/test/{d}_single.json")).items()]
entity_KB = get_global_entity_MWOZ()

def score_MWOZ(model,file_to_score,epoch,up_sampler=False,balance_sampler=False):
    genr_json = json.load(open(file_to_score))
    GOLD, GENR, GOLD_API, GENR_API = [], [],[], []
    acc_API = []
    F1_score = []
    # F1_domain = {"train":[],"attraction":[],"hotel":[],"restaurant":[]}
    F1_domain = {"train":[],"attraction":[],"hotel":[],"restaurant":[],"taxi":[]}
    # F1_API_domain = {"train":[],"hotel":[],"restaurant":[],"taxi":[]}
    total_match = []
    total_success = []
    # match_dy_domain = {"train":[],"attraction":[],"hotel":[],"restaurant":[]}
    # success_dy_domain = {"train":[],"attraction":[],"hotel":[],"restaurant":[]}
    match_dy_domain = {"train":[],"attraction":[],"hotel":[],"restaurant":[],"taxi":[]}
    success_dy_domain = {"train":[],"attraction":[],"hotel":[],"restaurant":[],"taxi":[]}
    for uuid,v in tqdm(gold_json.items(), total=len(gold_json)):
        ## MATCH
        match = {}
        entity_row_match = {}
        for dom, goal in v["goal"].items():
            if dom in ["train","attraction","hotel","restaurant"]:
                if("info" in goal):
                    query = to_query(dom, goal['info'], ["trainId"] if dom == 'train' else ['name'])
                    database.execute(query)
                    all_rows = database.fetchall()
                    if(len(all_rows)>0):
                        entity_row_match[dom] = all_rows
                        match[dom] = 0 

        ## SUCCESS
        success = {}
        entity_row_success = {}
        for dom, goal in v["goal"].items():
            if(goal and 'reqt' in goal):
                goal['reqt'] = [e for e in goal['reqt'] if e in ['phone', 'address', 'postcode']]#,'trainId']]
                if len(goal['reqt'])>0 and dom in ["train","attraction","hotel","restaurant"]:
                    query = to_query(dom, goal['info'], goal['reqt'])
                    database.execute(query)
                    all_rows = database.fetchall()
                    if(len(all_rows)>0):
                        if(dom=="train" and "leaveAt" in goal['reqt']):
                            for i in range(len(all_rows)):
                                all_rows[i] = list(all_rows[i])
                                time = all_rows[i][goal['reqt'].index("leaveAt")]
                                mins=int(time%60)
                                hours=int(time/60)
                                if(len(str(hours)))==1: hours = "0"+str(hours)
                                if(len(str(mins)))==1: mins = "0"+str(mins)
                                all_rows[i][goal['reqt'].index("leaveAt")] = str(hours)+str(mins)
                        if(dom=="train" and "arriveBy" in goal['reqt']):
                            for i in range(len(all_rows)):
                                all_rows[i] = list(all_rows[i])
                                time = all_rows[i][goal['reqt'].index("arriveBy")]
                                mins=int(time%60)
                                hours=int(time/60)
                                if(len(str(hours)))==1: hours = "0"+str(hours)
                                if(len(str(mins)))==1: mins = "0"+str(mins)
                                all_rows[i][goal['reqt'].index("arriveBy")] = str(hours)+str(mins)
                        entity_row_success[dom] = list( list(r) for r in all_rows)
                        success[dom] = 0 


        if("train" in match):
            if("book" not in v["goal"]['train']):
                match["train"] = 1
        gen_sentence = genr_json[uuid.lower()]
        gold_sentence = [v['conversation'] for v in test if v['src']==uuid][0]
        domain_id = [v["domain"].replace("_single","") for v in test if v['src']==uuid][0]

        if domain_id == "taxi":
            # print(v["goal"]['taxi'])
            match["taxi"] = 1

        for gold in gold_sentence:
            if(gold['spk']=="API"):
                if(len(gold['text'].split())==3):
                    ref,_,_ = gold['text'].split()
                elif(len(gold['text'].split())==2):
                    ref,_ = gold['text'].split()
                else:
                    ref = gold['text']
                
                if(ref!= "none"):
                    # if(domain_id in entity_row_success):
                    #     for j in range(len(entity_row_success[domain_id])):
                    #         entity_row_success[domain_id][j].append(ref)
                    # else:
                    #     ref = gold['text']
                    
                    # if(ref!= "none"):
                    if(domain_id in entity_row_success):
                        for j in range(len(entity_row_success[domain_id])):
                            entity_row_success[domain_id][j].append(ref)
                    else:
                        entity_row_success[domain_id] = [[ref]]
                    success[domain_id] = 0

        gold_sentence = [g for g in gold_sentence if g['spk'] not in ["API","USR"]]
        for gen, gold in zip(gen_sentence,gold_sentence):
            assert gen['spk'] == gold['spk']
            if(gen['spk']=="SYS-API"):
                GOLD_API.append(gold['text'])
                GENR_API.append(gen['text'])
                if("name" in gold["text"] or "trainId" in gold["text"]):
                    gold_txt = gold["text"].replace(" ","").replace("the","")
                    gen_txt = gen["text"].replace(" ","").replace("the","")

                    gold_map = gold_txt.replace("'","' ").replace("= '", "='").split(" ")
                    gen_map = gen_txt.replace("'","' ").replace("= '", "='").split(" ")

                    gold_map.sort()
                    gen_map.sort()

                    if gold_map == gen_map:
                        acc_API.append(1)
                    else:
                        acc_API.append(0)

                    # print(">", gold["text"].replace(" ",""))
                    # print(gen["text"].replace(" ",""))
                    # print()
                    # if(gold["text"].replace(" ","").replace("the","") == gen["text"].replace(" ","").replace("the","")):
                    #     acc_API.append(1)
                    # else:
                    #     acc_API.append(0)
            else:
                GOLD.append(gold['text'])
                GENR.append(align_GPT2(gen['text']))

                F1, count = compute_prf(align_GPT2(gen['text']), gold['entities'][domain_id], entity_KB)
                if(count==1):
                    F1_score.append(F1)
                    F1_domain[domain_id].append(F1)
                # input()
        # input()

        

        match_score = 0
        success_score = 0
        ## match_score 
        for k_dom,v_entities in entity_row_match.items():
            for row_table in v_entities:
                row_table = [e.lower() for e in row_table if e not in ["?","-"]]
                flag = [0 for _ in row_table]
                for idx_c, clmn_val in enumerate(row_table):
                    for sent in gen_sentence:
                        if(sent['spk'] == "SYS"):
                            if(clmn_val.lower() in sent['text'].lower()):
                                flag[idx_c] = 1 
                if(all(flag)): 
                    match[k_dom] = 1

        if(len(match)>0):
            match_score = int(all([int(v)==1 for k,v in match.items()]))
        else: 
            match_score = 1

        if(match_score==1):
            for k_dom,v_entities in entity_row_success.items():
                for row_table in v_entities:
                    row_table = [e.lower() for e in row_table if e not in ["?","-"]]
                    flag = [0 for _ in row_table]
                    for idx_c, clmn_val in enumerate(row_table):
                        for sent in gen_sentence:
                            if(sent['spk'] == "SYS"):
                                if(clmn_val.lower() in align_GPT2(sent['text']).lower()):
                                    flag[idx_c] = 1 
                    if(all(flag) and match[k_dom]): 
                        success[k_dom] = 1

            if(len(success)>0):
                success_score = int(all([int(v)==1 for k,v in success.items()]))
                # if(int(all([int(v)==1 for k,v in success.items()])) ==0 and domain_id =="taxi"):
                #     print(uuid)
            else:
                success_score = 1

        # print(match_score)
        # print(success_score)
        total_success.append(success_score) 
        success_dy_domain[domain_id].append(success_score)
        total_match.append(match_score) 
        match_dy_domain[domain_id].append(match_score)

    MATCH = sum(total_match)/float(len(total_match))
    SUCCESS = sum(total_success)/float(len(total_success))

    MATCH_BY_DOMAIN = {dom: (sum(arr)/float(len(arr)))*100 for dom, arr in match_dy_domain.items()}
    SUCCE_BY_DOMAIN = {dom: (sum(arr)/float(len(arr)))*100 for dom, arr in success_dy_domain.items()}
    MATCH_BY_DOMAIN["ALL"] = MATCH*100
    SUCCE_BY_DOMAIN["ALL"] = SUCCESS*100
    MATCH_BY_DOMAIN["Model"] = model
    SUCCE_BY_DOMAIN["Model"] = model
    BLEU = moses_multi_bleu(np.array(GENR),np.array(GOLD))
    BLEU_API = moses_multi_bleu(np.array(GENR_API),np.array(GOLD_API))
    BERTScore = huggingface_bertscore(GENR, GOLD,lang="en",model_type="bert-base-uncased",num_layers=9,batch_size=1)

    MATCH_BY_DOMAIN["Epoch"] = epoch
    MATCH_BY_DOMAIN["Up"] = up_sampler
    MATCH_BY_DOMAIN["Balance"] = balance_sampler

    SUCCE_BY_DOMAIN["Epoch"] = epoch
    SUCCE_BY_DOMAIN["Up"] = up_sampler
    SUCCE_BY_DOMAIN["Balance"] = balance_sampler

    return {"Model":model,
            "Epoch": epoch,
            "Up": up_sampler,
            "Balance": balance_sampler,
            "BLEU":BLEU, 
            "BERTScore": BERTScore,
            "MATCH": MATCH *100,
            "SUCCS": SUCCESS *100,
            "F1":100*np.mean(F1_score), 
            "Train":100*np.mean(F1_domain["train"]), 
            "Attra":100*np.mean(F1_domain["attraction"]), 
            "Hotel":100*np.mean(F1_domain["hotel"]),
            "Restu":100*np.mean(F1_domain["restaurant"]),
            "Taxi":100*np.mean(F1_domain["taxi"]),
            # "BLEU API":BLEU_API,
            "ACC API": 100* np.mean(acc_API),
            # "F1 API":100*np.mean(F1_API_score)
            },MATCH_BY_DOMAIN,SUCCE_BY_DOMAIN



rows = [
{"Model":"Mem2Seq", "Epoch": None, "Up":None, "Balance":None, "BLEU":6.6, "MATCH":-1, "SUCCS":None, "BERTScore":None, "F1":21.62, "Train":None, "Attra":22.0, "Hotel":21.0, "Restu":22.4, "Taxi":None, "ACC API":None},
{"Model":"DSR", "Epoch": None, "Up":None, "Balance":None, "BLEU":9.1, "MATCH":-1, "SUCCS":None, "BERTScore":None,"F1":30.0, "Train":None, "Attra":28.0, "Hotel":27.1, "Restu":33.4, "Taxi":None, "ACC API":None},
{"Model":"GLMP", "Epoch": None, "Up":None, "Balance":None, "BLEU":6.9, "MATCH":-1, "SUCCS":None, "BERTScore":None,"F1":32.4, "Train":None, "Attra":24.4, "Hotel":28.1, "Restu":38.4, "Taxi":None, "ACC API":None},
{"Model":"DFF", "Epoch": None, "Up":None, "Balance":None, "BLEU":9.4, "MATCH":-1, "SUCCS":None, "BERTScore":None,"F1":35.1, "Train":None, "Attra":28.1, "Hotel":30.6, "Restu":40.9, "Taxi":None, "ACC API":None},
{"Model":"DAMN", "Epoch": None, "Up":None, "Balance":None, "BLEU":13.5, "MATCH":85.40, "SUCCS":70.40, "BERTScore":None,"F1":None, "Train":None, "Attra":None, "Hotel":None, "Restu":None, "Taxi":None, "ACC API":None},
{"Model":"GOLD", "Epoch": None, "Up":None, "Balance":None, "BLEU":None, "MATCH":87.17, "SUCCS":85.84, "BERTScore":None,"F1":None, "Train":None, "Attra":None, "Hotel":None, "Restu":None, "Taxi":None, "ACC API":None},
]
rows_match = []
rows_succs = []
for f in glob.glob("runs/*"):
    if("MWOZ_SINGLE" in f and os.path.isfile(f+'/result.json')):
        params = f.replace("MWOZ_SINGLE","MWOZSINGLE").split("/")[1].split("_")

        balance_sampler = False
        up_sampler = False
        if len(params) == 28:
            d_name,model,_,graph,_,adj,_,edge,_,unilm,_,flattenKB,_,hist_L,_,LR,_,epoch,_,wt,_,kb,_,lay,_,balance_sampler,_,up_sampler = params
        elif len(params) == 26:
            d_name,model,_,graph,_,adj,_,edge,_,unilm,_,flattenKB,_,hist_L,_,LR,_,epoch,_,wt,_,kb,_,lay,_,balance_sampler = params
        else:
            d_name,model,_,graph,_,adj,_,edge,_,unilm,_,flattenKB,_,hist_L,_,LR,_,epoch,_,wt,_,kb,_,lay = params
        st = model.upper()
        
        # st += f" {int(kb)}"
        st += f" {kb}"
        stats, match, succ = score_MWOZ(st,f+'/result.json', epoch, up_sampler, balance_sampler)
        rows.append(stats)
        rows_match.append(match)
        rows_succs.append(succ)
        # exit()
# rows.append(score_MWOZ("KB0",'results_KB0.json'))
# rows.append(score_MWOZ("KB50",'results_KB50.json'))
rows = sorted(rows, key=lambda i:i['MATCH'])
rows_match = sorted(rows_match, key=lambda i:i['ALL'])
rows_succs = sorted(rows_succs, key=lambda i:i['ALL'])

print(tabulate(rows,headers="keys",tablefmt='latex',floatfmt=".2f",numalign="center"))
print(tabulate(rows_match,headers="keys",tablefmt='simple',floatfmt=".2f"))
print(tabulate(rows_succs,headers="keys",tablefmt='simple',floatfmt=".2f"))