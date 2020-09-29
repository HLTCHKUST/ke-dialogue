import json
import os.path
from utils.eval_metrics import moses_multi_bleu
import glob as glob
import numpy as np
import jsonlines
from tabulate import tabulate
from tqdm import tqdm

def compute_prf_SMD(gold, pred, global_entity_list):#, kb_plain=None):
    # local_kb_word = [k[0] for k in kb_plain]
    TP, FP, FN = 0, 0, 0
    if len(gold)!= 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in set(pred):
            if p in global_entity_list:# or p in local_kb_word:
                if p not in gold:
                    FP += 1
                    print(p)
        # print("TP",TP)
        # print("FP",FP)
        # print("FN",FN)
        precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
        # print(precision)
        recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
        # print(recall)
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        # print(F1)
    else:
        precision, recall, F1, count = 0, 0, 0, 0
    return F1, count

def get_global_entity_KVR():
    with open('data/SMD/kvret_entities.json') as f:
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

def score_SMD(model,file_to_score):
    genr_json = json.load(open(file_to_score))
    global_entity_list = get_global_entity_KVR()
    GOLD, GENR = [], []
    F1_score = []
    F1_domain = {"navigate":[],"weather":[],"schedule":[]}
    num_lines = sum(1 for line in open("data/SMD/test.txt",'r'))
    with open("data/SMD/test.txt",'r') as f:
        idd = 0
        for line in tqdm(f,total=num_lines):
            line = line.strip()
            if line:
                if '#' in line:
                    idd += 1
                    i = 0
                    line = line.replace("#","")
                    task_type = line
                    continue

                _, line = line.split(' ', 1)
                if '\t' in line:
                    u, r, gold_ent = line.split('\t')
                    # print(gold_ent, genr_json[str(idd)][i]['text'].split(" "))
                    F1, count = compute_prf_SMD(eval(gold_ent), genr_json[str(idd)][i]['text'].replace("."," .").replace("'"," '").replace("?"," ?").replace(","," ,").replace("!"," !").replace("  "," ").split(" "), global_entity_list)
                    if(count==1): 
                        F1_score.append(F1)
                        F1_domain[task_type].append(F1)
                    generated = genr_json[str(idd)][i]['text']
                    # print("USER",u)
                    # print("GOLD",r)
                    # print("GENE",generated)
                    GOLD.append(r)
                    GENR.append(generated.replace(",",""))
                    # print("ENTI",eval(gold_ent))
                    # print(F1)
                    # print()
                    # print()
                    # print()
                    i += 1
                    # input()

    BLEU = moses_multi_bleu(np.array(GENR),np.array(GOLD))

    return {"Model":model,
            "BLEU":BLEU, 
            "F1":100*np.mean(F1_score), 
            "F1 navigate":100*np.mean(F1_domain["navigate"]), 
            "F1 weather":100*np.mean(F1_domain["weather"]), 
            "F1 schedule":100*np.mean(F1_domain["schedule"])}

rows = [
        {"Model":"KVRet","BLEU":13.2,"F1":48.0,"F1 navigate":44.5,"F1 weather":53.3,"F1 schedule":62.9},  
        {"Model":"MLMN","BLEU":17.1,"F1":55.1,"F1 navigate":41.3,"F1 weather":47.0,"F1 schedule":68.3},
        # {"Model":"---","BLEU":"---","F1":"---","F1 navigate":"---","F1 weather":"---","F1 schedule":"---"},
        {"Model":"Mem2Seq","BLEU":12.2,"F1":33.40,"F1 navigate":20.00,"F1 weather":49.30,"F1 schedule":32.80},
        {"Model":"KBRet","BLEU":13.9,"F1":53.7,"F1 navigate":54.5,"F1 weather":52.2,"F1 schedule":55.6},
        {"Model":"GLMP","BLEU":13.9,"F1":60.7,"F1 navigate":54.6,"F1 weather":56.5,"F1 schedule":72.5},
        {"Model":"DFF","BLEU":14.4,"F1":62.7,"F1 navigate":57.9,"F1 weather":57.6,"F1 schedule":73.1},
        ]

# for f in glob.glob("runs/*"):
#     if("SMD" in f and os.path.isfile(f+'/result.json')):
#         d_name,model,_,graph,_,adj,_,edge,_,unilm,_,flattenKB,_,hist_L,_,*_ = f.split("/")[1].split("_")
#         st = model
#         if(eval(graph)): st+= "+NODE "
#         if(eval(adj)): st+= "+ADJ "
#         if(eval(edge)): st+= "+EDGES "
#         if(eval(unilm)): st+= "+UNI "
#         if(eval(flattenKB)): st+= "+KB "
#         rows.append(score_SMD(st,f+'/result.json'))

# baseline
rows.append({"Model":"gpt2","BLEU":14.72,"F1":39.11,"F1 navigate":23.41,"F1 weather":53.74,"F1 schedule":52.26})

# previous results
rows.append({"Model":"gpt2+KB","BLEU":14.72,"F1":57.50,"F1 navigate":49.00,"F1 weather":58.63,"F1 schedule":71.13})
rows.append({"Model":"T10","BLEU":11.24,"F1":52.88,"F1 navigate":50.26,"F1 weather":51.64,"F1 schedule":58.62})
rows.append({"Model":"T25","BLEU":12.26,"F1":55.00,"F1 navigate":50.46,"F1 weather":52.91,"F1 schedule":64.87})
rows.append({"Model":"T50","BLEU":13.01,"F1":56.43,"F1 navigate":50.04,"F1 weather":54.25,"F1 schedule":69.60})
rows.append({"Model":"T75","BLEU":13.67,"F1":58.79,"F1 navigate":52.56,"F1 weather":56.39,"F1 schedule":71.89})
rows.append({"Model":"T100","BLEU":0,"F1":0,"F1 navigate":53.53,"F1 weather":0,"F1 schedule":72.58})

rows.append(score_SMD("GPT2+FINE",'runs/SMD_gpt2_graph_False_adj_False_edge_False_unilm_False_flattenKB_False_historyL_1000000000_lr_6.25e-05_epoch_10_weighttie_False_kbpercentage_0_layer_12/all_results.json'))

print(tabulate(rows,headers="keys",tablefmt='latex',floatfmt=".2f"))
