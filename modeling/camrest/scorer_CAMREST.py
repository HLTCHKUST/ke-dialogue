import os, sys
sys.path.append('../..')

import json
import os.path
from utils.eval_metrics import moses_multi_bleu, huggingface_bertscore
import glob as glob
import numpy as np
import jsonlines
from tabulate import tabulate
import json
import matplotlib
import matplotlib.pyplot as plt
import re

def compute_prf(pred, gold, global_entity_list):
    TP, FP, FN = 0, 0, 0
    if len(gold)!= 0:
        cnt = 1
        for g in gold:
            if g.lower() in pred:
                TP += 1
            else:
                FN += 1
        for p in set(pred):
            if p in global_entity_list:
                if p not in gold:
                    FP += 1
        precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
        recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
    else:
        F1 = 0
        cnt = 0
    return F1, cnt

def entityList(): 
    KB = json.load(open("data/CamRest/KB.json"))
    glob = set()
    for item in KB:
        for k, v in item.items():
            if(v!= "restaurant"):
                if(k =="postcode"):
                    glob.add(v.replace(".","").replace(",","").replace(" ","").lower())   
                else: 
                    glob.add(v.replace(" ","_").lower())
    glob = list(glob)
    return glob

def score_MM(model,file_to_score,flattenKB=False):
    genr_json = json.load(open(file_to_score))
    global_ent = entityList()
    GOLD, GENR = [], []
    F1_score = []
    for ids, val in genr_json.items():
        if("api_call" not in val["gold"]):
            gold_ent = get_entity(global_ent, val["gold"].strip().lower().replace(".",""))
            pred = val["resp"].strip().lower().replace(".","").split(" ")
            F1, cnt = compute_prf(pred, gold_ent, global_ent)
            if(cnt!=0): 
                F1_score.append(F1)
            GOLD.append(val["gold"].strip().lower())
            GENR.append(val["resp"].strip().lower())

    BLEU = moses_multi_bleu(np.array(GENR),np.array(GOLD))
    BERTScore = huggingface_bertscore(GENR,GOLD,lang="en",model_type="bert-base-uncased",num_layers=9,batch_size=1)
#     BLEURT = google_bleurt(GENR,GOLD)
    return {
            "Model":model,
            "KB":0,
            "BLEU": BLEU,
            "BERTScore": BERTScore,
#             "BLEURT": BLEURT,
            "F1": 100*np.mean(F1_score)
            }

def score_KBRet(model,file_to_score,file_to_gold,flattenKB=False):
    predfile = open(file_to_score, "r")
    goldfile = open(file_to_gold, "r")
    def entityList(): 
        KB = json.load(open("data/CamRest/KB.json"))
        glob = set()
        for item in KB:
            for k, v in item.items():
                if(v!= "restaurant"):
                    glob.add(v.replace(" ","_").lower())
            
        glob = list(glob)
        return glob

    global_ent = entityList()
    GOLD, GENR = [], []
    F1_score = []
    for pred, gold in zip(predfile,goldfile):
        # print(gold.strip().lower())
        # print(pred.strip().lower())
        gold_ent = get_entity(global_ent, gold.strip().lower())
        pred_ = pred.strip().lower().split(" ")
        F1, cnt = compute_prf(pred_, gold_ent, global_ent)
        if(cnt!=0): 
            # print(gold_ent)
            F1_score.append(F1)
        #     print(F1)
        # print()
        GOLD.append(gold.strip().lower())
        GENR.append(pred.strip().lower())

    BLEU = moses_multi_bleu(np.array(GENR),np.array(GOLD))
    BERTScore = huggingface_bertscore(GENR,GOLD,lang="en",model_type="bert-base-uncased",num_layers=9,batch_size=1)
#     BLEURT = google_bleurt(GENR,GOLD)
    return {
            "Model":model,
            "KB":0,
            "BLEU": BLEU,
            "BERTScore": BERTScore,
#             "BLEURT": BLEURT,
            "F1": 100*np.mean(F1_score)
            }


def score_BABI(model,file_to_score,flattenKB=False,kb=0):
    gold_file = 'data/CamRest/test.txt'
    genr_json = json.load(open(file_to_score))
    global_ent = entityList()


    turn_acc = []
    dial_acc = []
    acc_temp = []
    GOLD, GENR = [], []
    uGOLD, uGENR = [], []
    F1_score = []
    with open(gold_file,'r') as f:
        idd = 0
        j = 0
        for line in f:
            if(line == "\n"):
                turn_acc += acc_temp
                if(all(ele == 1 for ele in acc_temp)):
                    dial_acc.append(1)
                else: 
                    dial_acc.append(0)
                idd += 1
                j = 0 
                acc_temp = []
            else:
                _, line = line.replace("\n","").split(' ', 1)
                if ("\t" in line):
                    _, syst = line.split("\t")
                    if("i'm on it" not in syst and "api_call" not in syst and "ok let me look into some options for you" not in syst):
                        assert genr_json[str(idd)][j]["spk"] == "SYS"
                        gold_ent = get_entity(global_ent, syst.strip().lower().replace(".",""))
                        pred = genr_json[str(idd)][j]['text'].strip().lower().replace("."," .").replace("'"," '").replace("?"," ?").replace(","," ,").replace("!"," !").replace("  "," ").split(" ")

                        F1, cnt = compute_prf(pred, gold_ent, global_ent)
                        if(cnt!=0): 
                            F1_score.append(F1)
                        GOLD.append(syst.strip().lower())
                        GENR.append(genr_json[str(idd)][j]['text'].strip().lower().replace("."," .").replace("'"," '").replace("?"," ?").replace(","," ,").replace("!"," !").replace("  "," "))
                        
                        uGOLD.append(re.sub(r' +', ' ', syst))
                        uGENR.append(re.sub(r' +', ' ', genr_json[str(idd)][j]['text'].strip().lower().replace("."," .").replace("'"," '").replace("?"," ?").replace(","," ,").replace("!"," !").replace("  "," ")))

                        if(genr_json[str(idd)][j]['text'].strip().lower() == syst.strip().lower()):
                            acc_temp.append(1)
                        else:
                            acc_temp.append(0)
                        j += 1
                    else:
                        if(flattenKB): j += 1

    BLEU = moses_multi_bleu(np.array(GENR),np.array(GOLD))
    BERTScore = huggingface_bertscore(uGENR, uGOLD,lang="en",model_type="bert-base-uncased",num_layers=9,batch_size=1)
    BLEURT = google_bleurt(GENR,GOLD)
    return {
            "Model":model,
            "KB":kb,
            # "ACC":100*np.mean(turn_acc), 
            # "DLG ACC": 100*np.mean(dial_acc),
            "BLEU": BLEU,
            "BERTScore": BERTScore,
            "BLEURT": BLEURT,
            "F1": 100*np.mean(F1_score)
            }

def get_entity(KB, sentence):
    list_entity = []
    for key in sentence.split(' '):
        if(key in KB):
            list_entity.append(key)
    return list_entity

rows_BABI = []

for f in glob.glob("runs/*"):
    print(f)
    if("CAMREST" in f and os.path.isfile(f+'/result.json')):
        params = f.split("/")[1].split("_")

        balance_sampler = False
        if len(params) == 26:
            d_name,model,_,graph,_,adj,_,edge,_,unilm,_,flattenKB,_,hist_L,_,LR,_,epoch,_,wt,_,kb,_,lay,_,balance_sampler = params
        else:
            d_name,model,_,graph,_,adj,_,edge,_,unilm,_,flattenKB,_,hist_L,_,LR,_,epoch,_,wt,_,kb,_,lay = params

        st = model
        if(eval(graph)): st+= "+NODE "
        if(eval(adj)): st+= "+ADJ "
        if(eval(edge)): st+= "+EDGES "
        if(eval(unilm)): st+= "+UNI "
        if(eval(flattenKB)): st+= "+REALK "
        # else:
            # st+= f"%KB={kb} "
            # # st+= f"+E{epoch} "
            # if(int(kb)!= 0):
            # else:
            #     kb =0 
        rows_BABI.append(score_BABI(st,f+'/result.json',eval(flattenKB),kb))

# rows_BABI.append(score_MM("Multi-level memory",'runs/MM/results.json',False))
# rows_BABI.append(score_KBRet("KBRet",'runs/KBret/pred.txt','runs/KBret/gold.txt',False))

print(tabulate(rows_BABI,headers="keys",tablefmt='latex',floatfmt=".3f",numalign="center"))