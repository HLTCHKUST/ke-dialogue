import json
import os.path
from utils.eval_metrics import moses_multi_bleu
import glob as glob
import numpy as np
import jsonlines
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate

def get_type_dict(kb_path, dstc2=False): 
    """
    Specifically, we augment the vocabulary with some special words, one for each of the KB entity types 
    For each type, the corresponding type word is added to the candidate representation if a word is found that appears 
    1) as a KB entity of that type, 
    """
    type_dict = {'R_restaurant':[]}

    kb_path_temp = kb_path
    fd = open(kb_path_temp,'r') 

    for line in fd:
        if dstc2:
            x = line.replace('\n','').split(' ')
            rest_name = x[1]
            entity = x[2]
            entity_value = x[3]
        else:
            x = line.split('\t')[0].split(' ')
            rest_name = x[1]
            entity = x[2]
            entity_value = line.split('\t')[1].replace('\n','')
    
        if rest_name not in type_dict['R_restaurant']:
            type_dict['R_restaurant'].append(rest_name)
        if entity not in type_dict.keys():
            type_dict[entity] = []
        if entity_value not in type_dict[entity]:
            type_dict[entity].append(entity_value)

    return type_dict

def entityList(kb_path, task_id):
    type_dict = get_type_dict(kb_path, dstc2=(task_id==6))
    entity_list = []
    for key in type_dict.keys():
        for value in type_dict[key]:
            entity_list.append(value)
    return entity_list


def compute_prf(pred, gold, global_entity_list):
    TP, FP, FN = 0, 0, 0
    if len(gold)!= 0:
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
        F1 = None
    return F1


def score_BABI(model,file_to_score,OOV,layer,KB):
    if(OOV):
        gold_file = 'data/dialog-bAbI-tasks/dialog-babi-task5tst-OOV.txt'
    else:
        gold_file = 'data/dialog-bAbI-tasks/dialog-babi-task5tst.txt'
    genr_json = json.load(open(file_to_score))
    turn_acc = []
    dial_acc = []
    acc_temp = []
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
                        # print(genr_json[str(idd)][j]['text'].strip(),syst.strip())
                        if(genr_json[str(idd)][j]['text'].strip() == syst.strip()):
                            acc_temp.append(1)
                        else:
                            acc_temp.append(0)
                        j += 1
    return {
            "Model":model,
            # "Layer": layer, 
            "KB":KB,
            "ACC":100*np.mean(turn_acc), 
            "DLG ACC": 100*np.mean(dial_acc)
            }

def get_entity(KB, sentence):
    list_entity = []
    for key in sentence.split(' '):
        if(key in KB):
            list_entity.append(key)
    return list_entity

def score_DSTC2(model,file_to_score):
    global_ent = entityList('data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-kb.txt',6)
    # print(global_ent)
    gold_file = 'data/dialog-bAbI-tasks/dialog-babi-task6tst.txt'
    genr_json = json.load(open(file_to_score))
    turn_acc = []
    dial_acc = []
    acc_temp = []
    F1_score = []
    GOLD, GENR = [], []
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
                    if("i'm on it" not in syst and "api_call" not in syst and "ok let me look into some options for you" not in syst and "Hello , welcome to the Cambridge restaurant system . You can ask for restaurants by area , price range or food type . How may I help you ?" not in syst):
                        assert genr_json[str(idd)][j]["spk"] == "SYS"
                        ## FOR BLUE SCORE
                        GOLD.append(syst)
                        GENR.append(genr_json[str(idd)][j]['text'].strip())
                        ## FOR ENTITY F1
                        gold_ent = get_entity(global_ent, syst)
                        pred = genr_json[str(idd)][j]['text'].strip().split(" ")
                        F1 = compute_prf(pred, gold_ent, global_ent)
                        if(F1): 
                            F1_score.append(F1)

                        ### DIALGOUE ACCURACY and ACCURACY 
                        if(genr_json[str(idd)][j]['text'].strip() == syst.strip()):
                            acc_temp.append(1)
                        else:
                            acc_temp.append(0)
                        j += 1


    BLEU = moses_multi_bleu(np.array(GENR),np.array(GOLD))
    return {
            "Model":model,
            "ACC":100*np.mean(turn_acc), 
            "DLG ACC": 100*np.mean(dial_acc),
            "BLEU": BLEU,
            "F1": 100*np.mean(F1_score)
            }

rows_BABI = [
    {"Model":"QRN","KB":None,"ACC":99.6,"DLG ACC":None},
    {"Model":"QRN OOV","KB":None,"ACC":67.8,"DLG ACC":None},
    {"Model":"Mem2Seq","KB":None,"ACC":97.9,"DLG ACC":69.6},
    {"Model":"Mem2Seq OOV","KB":None,"ACC":84.5,"DLG ACC":2.3},
    {"Model":"GLMP","KB":None,"ACC":99.2,"DLG ACC":88.5},
    {"Model":"GLMP OOV","KB":None,"ACC":92.0,"DLG ACC":21.7},
]

for f in glob.glob("runs/*"):
    if("BABI" in f and os.path.isfile(f+'/result.json')):
        params = f.split("/")[1].split("_")

        balance_sampler = False
        if len(params) == 26:
            d_name,model,_,graph,_,adj,_,edge,_,unilm,_,flattenKB,_,hist_L,_,LR,_,epoch,_,wt,_,kb,_,lay,_,balance_sampler = params
        else:
            d_name,model,_,graph,_,adj,_,edge,_,unilm,_,flattenKB,_,hist_L,_,LR,_,epoch,_,wt,_,kb,_,lay = params


        st = model.replace("gpt2-bAbI","GPT2")
        # if(eval(graph)): st+= "+NODE "
        # if(eval(adj)): st+= "+ADJ "
        # if(eval(edge)): st+= "+EDGES "
        # if(eval(unilm)): st+= "+UNI "
        # if(eval(flattenKB)): st+= "+KB "
        # if(eval(lay)): st+= f"+L {lay}"
        # # st+= f"+E{epoch} "
        # st+= f"+KB {kb} "
        rows_BABI.append(score_BABI(st,f+'/result.json',OOV=False,layer=lay,KB=kb))
        rows_BABI.append(score_BABI(st+" OOV",f+'/result_OOV.json',OOV=True,layer=lay,KB=kb))
    # if("DSTC" in f and os.path.isfile(f+'/result.json')):
    #     d_name,model,_,graph,_,adj,_,edge,_,unilm,_,flattenKB,_,hist_L,_,LR,_,epoch,*_ = f.split("/")[1].split("_")
    #     st = model
    #     if(eval(graph)): st+= "+NODE "
    #     if(eval(adj)): st+= "+ADJ "
    #     if(eval(edge)): st+= "+EDGES "
    #     if(eval(unilm)): st+= "+UNI "
    #     if(eval(flattenKB)): st+= "+KB "
    #     st+= f"+E{epoch} "
    #     rows_DSTC.append(score_DSTC2(st,f+'/result.json'))
print(tabulate(rows_BABI,headers="keys",tablefmt='latex',floatfmt=".2f",numalign="center"))
# print(tabulate(rows_DSTC,headers="keys",tablefmt='simple',floatfmt=".2f"))

# rows_BABI = sorted(rows_BABI, key = lambda r: int(r["KB"]))
# x = [int(r["KB"]) for r in rows_BABI if "OOV" not in r["Model"] and int(r["Layer"])==12]
# x_OOV = [int(r["KB"]) for r in rows_BABI if "OOV" in r["Model"] and int(r["Layer"])==12]

# y = [float(r["ACC"]) for r in rows_BABI if "OOV" not in r["Model"] and int(r["Layer"])==12]
# y_OOV = [float(r["ACC"]) for r in rows_BABI if "OOV" in r["Model"] and int(r["Layer"])==12]
# fig, ax = plt.subplots()
# ax.plot(x, y,label="GPT2")
# ax.plot(x_OOV, y_OOV,label="GPT2+OOV")
# plt.axhline(y=99.2,label="GLMP", color='r', linestyle='--')
# plt.axhline(y=92.0,label="GLMP+OOV", color='g', linestyle='--')
# # plt.axhline(y=99.66,label="QRN", color='y', linestyle='--')
# # plt.axhline(y=67.8,label="QRN+OOV", color='b', linestyle='--')
# ax.set(xlabel='Template', ylabel='Accuracy',
#        title='Template Effectiveness')
# ax.grid()
# ax.legend()
# fig.savefig("TEMPvsACC.png")

# ### Layers vs ACC
# rows_BABI = sorted(rows_BABI, key = lambda r: int(r["Layer"]))
# x = [int(r["Layer"]) for r in rows_BABI if "OOV" not in r["Model"] and int(r["KB"])==264]
# x_OOV = [int(r["Layer"]) for r in rows_BABI if "OOV" in r["Model"] and int(r["KB"])==264]

# y = [float(r["ACC"]) for r in rows_BABI if "OOV" not in r["Model"] and int(r["KB"])==264]
# y_OOV = [float(r["ACC"]) for r in rows_BABI if "OOV" in r["Model"] and int(r["KB"])==264]
# fig, ax = plt.subplots()
# ax.plot(x, y,label="GPT2")
# ax.plot(x_OOV, y_OOV,label="GPT2+OOV")


# ax.set(xlabel='Layers', ylabel='Accuracy',
#        title='Template Effectiveness')
# ax.grid()
# ax.legend()

# fig.savefig("LAYEvsACC.png")