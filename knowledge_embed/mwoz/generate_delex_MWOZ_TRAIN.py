import json
import os.path
import glob as glob
import numpy as np
import jsonlines
from tabulate import tabulate
import re
from tqdm import tqdm
import sqlite3
import editdistance
from collections import defaultdict
import pprint
from itertools import product
from copy import deepcopy

def set_seed(val):
    np.random.seed(val)

set_seed(42)

timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat = re.compile("\d{1,3}[.]\d{1,2}")


fin = open("mapping.pair","r")
replacements = []
for line in fin.readlines():
    tok_from, tok_to = line.replace('\n', '').split('\t')
    replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def normalize(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)
    text = re.sub(r"gueshouses", "guesthouse", text)
    text = re.sub(r"guest house", "guesthouse", text)
    text = re.sub(r"rosas bed and breakfast", "rosa s bed and breakfast", text)
    text = re.sub(r"el shaddia guesthouse", "el shaddai", text)
    

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    # text = re.sub(timepat, ' [value_time] ', text)
    # text = re.sub(pricepat, ' [value_price] ', text)
    #text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\":\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text

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

def substringSieve(string_list):
    string_list.sort(key=lambda s: len(s), reverse=True)
    out = []
    for s in string_list:
        if not any([s in o for o in out]):
            out.append(s)
    return out

def to_query(domain, dic, reqt):
    if reqt:
        q = f"SELECT {','.join(reqt)} FROM {domain} where"
    else:
        q = f"SELECT * FROM {domain} where"
    for k,v in dic.items():
        if v == "swimmingpool": v = "swimming pool"
        if v == "nightclub": v = "night club"
        if v == "the golden curry": v = "golden curry"
        if v == "mutliple sports": v = "multiple sports"
        if v == "the cambridge chop house": v = "cambridge chop house"
        if v == "the fitzwilliam museum": v = "fitzwilliam museum"
        if v == "the good luck chinese food takeaway": v = "good luck chinese food takeaway"
        if v == "the cherry hinton village centre": v = "cherry hinton village centre"
        if v == "the copper kettle": v = "copper kettle"
        if v == "pizza express Fen Ditton": v = "pizza express"
        if v == "shiraz restaurant": v = "shiraz"
        if v == "christ's college": v = "christ college"
        if v == "good luck chinese food takeaway": v = "chinese"

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

def convert_time_int_to_time(all_rows,clmn):#leaveAt_id,arriveBy_id):
    leaveAt_id = -1
    arriveBy_id = -1
    if('leaveAt' in clmn):
        leaveAt_id = clmn.index('leaveAt')
    if('arriveBy' in clmn):
        arriveBy_id = clmn.index('arriveBy')
    if(leaveAt_id!= -1):
        for i in range(len(all_rows)):
            all_rows[i] = list(all_rows[i])
            time = int(all_rows[i][leaveAt_id])
            mins=int(time%60)
            hours=int(time/60)
            if(len(str(hours)))==1: hours = "0"+str(hours)
            if(len(str(mins)))==1: mins = "0"+str(mins)
            all_rows[i][leaveAt_id] = str(hours)+str(mins)
    if(arriveBy_id!= -1):
        for i in range(len(all_rows)):
            all_rows[i] = list(all_rows[i])
            time = int(all_rows[i][arriveBy_id])
            mins=int(time%60)
            hours=int(time/60)
            if(len(str(hours)))==1: hours = "0"+str(hours)
            if(len(str(mins)))==1: mins = "0"+str(mins)
            all_rows[i][arriveBy_id] = str(hours)+str(mins)
    return all_rows

def get_entity_by_type(info,clmn,post_fix="-info"):
    ### get goal information
    query = to_query("train", info, clmn)
    database.execute(query)
    all_rows = database.fetchall()
    all_rows = convert_time_int_to_time(all_rows,clmn)
    entity_by_type = {c+post_fix:set() for c in clmn}
    for rows in all_rows:
        for i,c in enumerate(clmn):
            entity_by_type[c+post_fix].add(rows[i])
    # entity_by_type["number_of_options"] = [len(all_rows)]
    return entity_by_type

def parse_results(dic_data,semi,domain):
    book_query = str(domain)
    if(domain == "taxi"):
        for k, t in semi.items():
            if k in ["leaveAt","destination","departure","arriveBy"]:
                book_query += f" {k} = '{normalize(t)}'"

    if(domain == "hotel"):
        if dic_data["day"]== "" or dic_data["stay"]== "" or dic_data["people"]== "":
            return None,None
    results = None
    if(len(dic_data['booked'])>0):
        if(domain == "train" and 'trainID' in dic_data['booked'][0]):
            book_query += f" trainID = '{normalize(dic_data['booked'][0]['trainID'])}'"
            results =  dic_data['booked'][0]['reference']
        elif(domain != "taxi" and 'name' in dic_data['booked'][0]):
            book_query += f" name = '{normalize(dic_data['booked'][0]['name'])}'"
            results =  dic_data['booked'][0]['reference']
        else:
            results =  dic_data['booked'][0]
    elif(domain == "hotel" and semi['name']!="not mentioned"):
        book_query += f" name = '{normalize(semi['name'])}'"

    for k, t in dic_data.items():
        if(k != 'booked'):
            book_query += f" {k} = '{normalize(t)}'"

    return book_query, results

def check_metadata(dic, state):
    for d,v in dic.items():
        if(state[d]==0 or state[d]!= v['book']['booked']):
            if(len(v['book']['booked'])>0):
                state[d] = v['book']['booked']
                return parse_results(v['book'],v['semi'],d), state 
            for k, v1 in v['book'].items():
                if(k != 'booked' and v1 != ""):
                    return parse_results(v['book'],v['semi'],d), state
    return (None, None), state

def get_booking_query(text):
    domain = {"global":set(),"train":[],"attraction":[],"hotel":[],"restaurant":[],"taxi":[],
              "police":[],"hospital":[],"generic":[]}
    domain[text.split()[0]] = re.findall(r"'(.*?)'", text)
    return domain

def delexer(turns,dictionary,entity_info):
    text_delex = normalize(turns['text'])
    ### first serch using SPEECH ACT
    for k,v in turns['dialog_act'].items():
        for [att,val] in v:
            if( (att not in ["none","Ref","People","Ticket"] and val not in ["-","?"]) or (k=="Train-Inform" and att=="Ticket") ):
                if(att in ["Leave","Arrive"]):
                    if( normalize(val).isdecimal() and len(normalize(val))==4):
                        dictionary[att.lower()].append(normalize(val))
                elif(att=="Ticket"):
                    dictionary[att.lower()].append(normalize(val).replace(" each","").replace(" per person",""))
                else:
                    dictionary[att.lower()].append(normalize(val))
                    if("street" in val.lower()):
                        dictionary[att.lower()].append(normalize(val).replace(" street",""))

    
    for k,v in entity_info.items():
        for val in v:
            if(type(val)==int and str(val) in text_delex):
                dictionary[k].append(str(val))
            else:
                if(normalize(val) in text_delex):
                    dictionary[k].append(normalize(val))
                elif("street" in val.lower() and normalize(val).replace(" street","") in text_delex):
                    dictionary[k].append(normalize(val).replace(" street",""))
                    turns['text'] = turns['text'].replace(normalize(val).replace(" street",""),normalize(val))


    return text_delex

def query_TRAINID_and_filter(entity_correct_train,r_delex_dictionary):
        # 'duration-correct', 'leaveAt-correct']
    if "leaveAt-info" in r_delex_dictionary or "leave" in r_delex_dictionary:
        if entity_correct_train['leaveAt-correct'] not in r_delex_dictionary.get("leaveAt-info",[]) and entity_correct_train['leaveAt-correct'] not in r_delex_dictionary.get("leave",[]):
            del entity_correct_train['leaveAt-correct']
        else:
            if(entity_correct_train['leaveAt-correct'] in r_delex_dictionary.get("leaveAt-info",[])):
                del r_delex_dictionary["leaveAt-info"][r_delex_dictionary["leaveAt-info"].index(entity_correct_train['leaveAt-correct'])]
            if(entity_correct_train['leaveAt-correct'] in r_delex_dictionary.get("leave",[])):
                del r_delex_dictionary["leave"][r_delex_dictionary["leave"].index(entity_correct_train['leaveAt-correct'])]
    else:
        del entity_correct_train['leaveAt-correct']

    if "arriveBy-info" in r_delex_dictionary or "arrive" in r_delex_dictionary:
        if entity_correct_train['arriveBy-correct'] not in r_delex_dictionary.get("arriveBy-info",[]) and entity_correct_train['arriveBy-correct'] not in r_delex_dictionary.get("arrive",[]):
            del entity_correct_train['arriveBy-correct']
        else:
            if(entity_correct_train['arriveBy-correct'] in r_delex_dictionary.get("arriveBy-info",[])):
                del r_delex_dictionary["arriveBy-info"][r_delex_dictionary["arriveBy-info"].index(entity_correct_train['arriveBy-correct'])]
            if(entity_correct_train['arriveBy-correct'] in r_delex_dictionary.get("arrive",[])):
                del r_delex_dictionary["arrive"][r_delex_dictionary["arrive"].index(entity_correct_train['arriveBy-correct'])]
    else:
        del entity_correct_train['arriveBy-correct']

    if "day-info" in r_delex_dictionary or "day" in r_delex_dictionary:
        if entity_correct_train['day-correct'] not in r_delex_dictionary.get("day-info",[]) and entity_correct_train['day-correct'] not in r_delex_dictionary.get("day",[]):
            del entity_correct_train['day-correct']
        else:
            if(entity_correct_train['day-correct'] in r_delex_dictionary.get("day-info",[])):
                del r_delex_dictionary["day-info"][r_delex_dictionary["day-info"].index(entity_correct_train['day-correct'])]
            if(entity_correct_train['day-correct'] in r_delex_dictionary.get("day",[])):
                del r_delex_dictionary["day"][r_delex_dictionary["day"].index(entity_correct_train['day-correct'])]
    else:
        del entity_correct_train['day-correct']

    if "departure-info" in r_delex_dictionary or "depart" in r_delex_dictionary:
        if entity_correct_train['departure-correct'] not in r_delex_dictionary.get("departure-info",[]) and entity_correct_train['departure-correct'] not in r_delex_dictionary.get("depart",[]):
            del entity_correct_train['departure-correct']
        else:
            if(entity_correct_train['departure-correct'] in r_delex_dictionary.get("departure-info",[])):
                del r_delex_dictionary["departure-info"][r_delex_dictionary["departure-info"].index(entity_correct_train['departure-correct'])]
            if(entity_correct_train['departure-correct'] in r_delex_dictionary.get("depart",[])):
                del r_delex_dictionary["depart"][r_delex_dictionary["depart"].index(entity_correct_train['departure-correct'])]
    else:
        del entity_correct_train['departure-correct']

    if "destination-info" in r_delex_dictionary or "dest" in r_delex_dictionary:
        if entity_correct_train['destination-correct'] not in r_delex_dictionary.get("destination-info",[]) and entity_correct_train['destination-correct'] not in r_delex_dictionary.get("dest",[]):
            del entity_correct_train['destination-correct']
        else:
            if(entity_correct_train['destination-correct'] in r_delex_dictionary.get("destination-info",[])):
                del r_delex_dictionary["destination-info"][r_delex_dictionary["destination-info"].index(entity_correct_train['destination-correct'])]
            if(entity_correct_train['destination-correct'] in r_delex_dictionary.get("dest",[])):
                del r_delex_dictionary["dest"][r_delex_dictionary["dest"].index(entity_correct_train['destination-correct'])]
    else:
        del entity_correct_train['destination-correct']

    if "ticket" in r_delex_dictionary:
        if entity_correct_train['price-correct'] not in r_delex_dictionary["ticket"]:
            del entity_correct_train['price-correct']
        else:
            del r_delex_dictionary["ticket"][r_delex_dictionary["ticket"].index(entity_correct_train['price-correct'])]
    else:
        del entity_correct_train['price-correct']

    if "time" in r_delex_dictionary:
        if entity_correct_train['duration-correct'] not in r_delex_dictionary["time"]:
            del entity_correct_train['duration-correct']
        else:
            del r_delex_dictionary["time"][r_delex_dictionary["time"].index(entity_correct_train['duration-correct'])]
    else:
        del entity_correct_train['duration-correct']

    if entity_correct_train['trainID-correct'] not in r_delex_dictionary["id"]:
        del entity_correct_train['trainID-correct']
    else:
        del r_delex_dictionary["id"][r_delex_dictionary["id"].index(entity_correct_train['trainID-correct'])]
    r_delex_dictionary = {k:v for k,v in r_delex_dictionary.items() if len(v)>0}
    return entity_correct_train,r_delex_dictionary


def get_name_hotel(conv,dict_delex):
    for conv_turn in reversed(conv):
        if "name" in dict_delex.keys():
            for ids_v, v in enumerate(r_delex_dictionary["name"]):
                if(v in conv_turn["text"]):
                    return v, ids_v
                if(v.replace("the ","") in conv_turn["text"]):
                    return v, ids_v
    return None, None

def get_trainID_train(conv,dict_delex):
    for conv_turn in reversed(conv):
        if "id" in dict_delex.keys():
            for ids_v, v in enumerate(r_delex_dictionary["id"]):
                if(v in conv_turn["text"]):
                    return v, ids_v
                if(v in conv_turn["text"]):
                    return v, ids_v
    return None, None

def get_start_end_ACT(ACT):
    dic = {}
    mapper = {"one":1,"two":2,"three":3,"3-star":3,"four":4,"five":5}
    for span in ACT:
        if(span[1]=="Stars"):
            if(span[2] in mapper.keys()):
                dic[mapper[span[2]]] = [span[3],span[4]]
            else:
                dic[span[2]] = [span[3],span[4]]
    return dic


pp = pprint.PrettyPrinter(indent=4)
conn = sqlite3.connect('MWOZ.db')
database = conn.cursor()
all_arriveBy = [r[0] for r in database.execute("SELECT DISTINCT arriveBy FROM train").fetchall()]
all_day = [r[0] for r in database.execute("SELECT DISTINCT day FROM train").fetchall()]
all_departure = [r[0] for r in database.execute("SELECT DISTINCT departure FROM train").fetchall()]
all_destination = [r[0] for r in database.execute("SELECT DISTINCT destination FROM train").fetchall()]
all_leaveAt = [r[0] for r in database.execute("SELECT DISTINCT leaveAt FROM train").fetchall()]
all_trainID = [r[0] for r in database.execute("SELECT DISTINCT trainID FROM train").fetchall()]

dialogue_mwoz = json.load(open("MultiWOZ_2.1/data.json"))
test_split = open("MultiWOZ_2.1/testListFile.txt","r").read()
val_split = open("MultiWOZ_2.1/valListFile.txt","r").read()
train, valid, test = get_splits(dialogue_mwoz,test_split,val_split)
split_by_single_and_domain = json.load(open("dialogue_by_domain.json"))

all_arriveBy_choice = []
all_leaveAt_choice = []
for k, dial in train.items():
    if(k.lower() in split_by_single_and_domain["train_single"]):
        goal = dial["goal"]['train']['info']
        if('leaveAt' in goal):
            all_leaveAt_choice.append(goal['leaveAt'])
        if('arriveBy' in goal):
            all_arriveBy_choice.append(goal['arriveBy'])
all_arriveBy_choice = list(set(all_arriveBy_choice))
all_leaveAt_choice = list(set(all_leaveAt_choice))

all_arriveBy_choice.sort()
all_leaveAt_choice.sort()
# print(all_leaveAt_choice)

def generate_all_query(r_delex_dictionary, entity_correct_train,info):
    contrains = [all_day,all_departure,all_destination,[None]]
    name = ['day','departure','destination']

    if('leaveAt' in info):
        contrains[3] = all_leaveAt_choice
        name.append('leaveAt')
    elif('arriveBy' in info):
        contrains[3] = all_arriveBy_choice
        name.append('arriveBy')
    clmn = [k.replace("-correct","")for k in entity_correct_train.keys()]
    lexicalized = []
    all_combo = list(product(*contrains))
    all_combo.sort()

    index = np.random.choice(len(all_combo), 500).tolist()
    list_combo = [ all_combo[indx] for indx in index ]
    for combo in list_combo:
        query = {name[i_c]:c for i_c, c in enumerate(combo)}
        database.execute(to_query("train", query, clmn))
        all_rows = database.fetchall()
        if(len(all_rows)>0):
            choice = str(len(all_rows))
            if('leaveAt' in entity_correct_train.keys()):
                min_time = min([int(row[clmn.index("leaveAt")]) for row in all_rows])
                all_rows = [ row for row in all_rows if int(row[clmn.index("leaveAt")])== min_time ]
            if('arriveBy' in entity_correct_train.keys()):
                max_time = max([int(row[clmn.index("arriveBy")]) for row in all_rows])
                all_rows = [ row for row in all_rows if int(row[clmn.index("arriveBy")])== max_time ]
            all_rows = convert_time_int_to_time(all_rows.copy(),clmn)

            for row in all_rows:
                results_correct = entity_correct_train.copy()
                r_dictionary = r_delex_dictionary.copy()
                for k in results_correct.keys():
                    results_correct[k] = row[clmn.index(k.replace("-correct",""))]
                if("choice" in r_dictionary):
                    r_dictionary["choice"] = choice
                if('leave' in r_dictionary):
                    r_dictionary["leave"] = normalize(combo[-1])
                elif('arrive' in r_dictionary):
                    r_dictionary["arrive"] = normalize(combo[-1])
                lexicalized.append([results_correct,r_dictionary])
    return lexicalized

clmn_train = ["trainID",'day','departure','destination','leaveAt']
cnt_single_entity_templates = []
al = 0
good = 0
count_flag = 0
data = []
skip1, skip2, skip3, skip4 = 0,0,0,0
for i, (k, dial) in enumerate(train.items()):
    if(k.lower() in split_by_single_and_domain["train_single"]):
        id_dialogue = k.lower()
        goal = dial["goal"]['train']
        # pp.pprint(goal)

        dictionary = defaultdict(list)
        
        if("trainID" in goal['info']): 
            dictionary["trainID"].append(normalize(goal['info']["trainID"]))
            
        entity_info = get_entity_by_type(goal['info'],list(goal['info'].keys()))
        conversation = []
        train_ID_BOOKED = ""
        state = {"train":0,"attraction":0,"hotel":0,"restaurant":0,"hospital":0,"police":0,"taxi":0,"bus":0}

        span_info_list = []
        for turns in dial["log"]:
            if(turns['metadata']):
                text_delex = delexer(turns,dictionary,entity_info)
                (book, results), state = check_metadata(turns['metadata'],state)
                if(book):
                    entities_by_domain_book = get_booking_query(book)
                    book_delex = book

                    conversation.append({"entities":entities_by_domain_book,"spk":"SYS-API","text":book,"text_delex":book_delex})
                    span_info_list.append(turns["span_info"])

                    dom_API = book.split()[0] ## first token is the API domain
                    
                    train_ID_BOOKED = book.split()[3].replace("'", "")

                    ## THIS IS A SIMULATION OF AN API RESULTS
                    if("dialog_act" in turns and dom_API == "train" and "Train-OfferBooked" in turns["dialog_act"]):
                        for elem_ in turns["dialog_act"]["Train-OfferBooked"]:
                            if(elem_[0]=="Ticket" and elem_[1] != "None"):
                                results = str(results)
                                results += " "+ str(elem_[1])
                    conversation.append({"spk":"API","text":str(results).lower(),"text_delex":str(results).lower()})
                    span_info_list.append(turns["span_info"])

                conversation.append({"spk":"SYS","text":normalize(turns['text']),"text_delex":text_delex,"span_info": turns["span_info"]})
                span_info_list.append(turns["span_info"])
            else:
                text_delex =  delexer(turns,dictionary,entity_info)

                conversation.append({"spk":"USR","text":normalize(turns['text']),"text_delex":text_delex,"span_info": turns["span_info"]})
                span_info_list.append(turns["span_info"])
        # for conv_turn in conversation:
        #     print(f"{conv_turn['spk']} >>> {conv_turn['text']}")


        r_delex_dictionary = {}
        len_r_DICT = {}
        # if(i == 487):
        #     print(dictionary.items())
        for key, d in dictionary.items():
            r_delex_dictionary[key] = list(set(substringSieve(d)))
            r_delex_dictionary[key].sort()
            len_r_DICT[key] = len(r_delex_dictionary[key])

            r_delex_dictionary[key].sort()
        
        al += 1
        train_id, idx_id = get_trainID_train(conversation,r_delex_dictionary)
        # print(train_id,idx_id)
        
        if(train_id==None or (train_id[0]!="t" and train_id[1]!="r") or train_id[2]==" " or len(train_id)!=6):
            skip1+=1
            continue

        # print(">", train_id.lower())
        if(train_ID_BOOKED!=""): 
            if train_ID_BOOKED.lower()!=train_id.lower(): 
                skip2+=1
                continue 
        # input()
        entity_correct_train = get_entity_by_type({"trainID":train_id.upper()},["arriveBy","day","departure","destination","duration","leaveAt","price","trainID"],"-correct")
        
        new_entity = {}
        for k,v in entity_correct_train.items():
            v = list(v)
            v.sort()
            new_entity[k] = v[0].lower()
        entity_correct_train = new_entity
        
        entity_correct_train, r_delex_dictionary = query_TRAINID_and_filter(entity_correct_train,r_delex_dictionary)

        total_lexer = {**{k:[v] for k,v in entity_correct_train.items()}, **r_delex_dictionary}
        if(len(r_delex_dictionary.keys())>2):
            skip3+=1
            continue
        if("leaveAt-correct" not in entity_correct_train.keys() and "arriveBy-correct" not in entity_correct_train.keys()):
            skip4+=1
            continue

        flag = True
        if("leave" in r_delex_dictionary and "choice" in r_delex_dictionary and "leaveAt" in goal['info']):
            flag = False
        if("arrive" in r_delex_dictionary and "choice" in r_delex_dictionary and "arriveBy" in goal['info']):
            flag = False
        if("leave" in r_delex_dictionary and len(r_delex_dictionary.keys())==1 and "leaveAt" in goal['info']):
            flag = False
        if("arrive" in r_delex_dictionary and len(r_delex_dictionary.keys())==1 and "arriveBy" in goal['info']):
            flag = False
        count_flag += 1
        if(flag):
            continue
        good += 1
        # print(entity_correct_train)
        # print(r_delex_dictionary)
        # r_delex_dictionary = {k:[v] for k,v in entity_correct_train.items()}
        lexicalized = generate_all_query(r_delex_dictionary, entity_correct_train, goal['info'])
        lexicalized = [{**{k:[v] for k,v in l[0].items()}, **l[1]} for l in lexicalized]
        # lexicalized = []
        # pp.pprint(total_lexer)
        # print()
        # print(len(conversation), len(span_info_list))
        rdelex_conv = []
        flag = True
        for i, conv_turn in enumerate(conversation):
            text_rdelex = conv_turn["text"]
            if conv_turn["spk"] in ["USR","SYS"]:
                if "trainID-correct" in total_lexer.keys():
                    for ids_v, v in enumerate(total_lexer["trainID-correct"]):
                        text_rdelex = text_rdelex.replace(v,f"[trainID-correct_{ids_v}]")
                        text_rdelex = text_rdelex.replace(v.replace("the ",""),f"[trainID-correct_{ids_v}]")
                for ty,val in total_lexer.items():
                    if(ty != "choice"):
                        for ids_v, v in enumerate(sorted(val, reverse=True, key=lambda item: len(item))):
                            text_rdelex = text_rdelex.replace(v,f"[{ty}_{ids_v}]")

                if "choice" in total_lexer.keys():
                    # print(">>>", span_info_list[i])
                    for info in span_info_list[i]:
                        if info[1] == "Choice":
                            # print(text_rdelex)
                            start_span, end_span = info[3], info[4]
                            value = info[4]
                            words = text_rdelex.split()
                            for t in range(len(words)):
                                if t == start_span:
                                    words[t] = '[choice_0]'
                                elif t > start_span and t <= end_span:
                                    words[t] = ''
                            # print(">before:" , text_rdelex)
                            text_rdelex = ' '.join(words)
                            # print(">after: ",text_rdelex)


                if("cambridge towninfo centre" not in text_rdelex and "towninfo centre" not in text_rdelex and "cambridge" not in text_rdelex):
                    # all_arriveBy
                    # all_day
                    # all_leaveAt

                    for day in all_day:
                        if(day in text_rdelex):
                            # print(day,text_rdelex)
                            flag = False
                            continue

                    for lat in all_leaveAt:
                        if(" "+str(lat) in text_rdelex):
                            # print(lat,text_rdelex)
                            flag = False
                            continue
                    for arb in all_arriveBy:
                        if(" "+str(arb) in text_rdelex):
                            # print(arb,text_rdelex)
                            flag = False
                            continue

                    for dest in all_destination:
                        if(" "+dest in text_rdelex):
                            # print(dest,text_rdelex)
                            flag = False
                            continue

                    for dpt in all_departure:
                        if(" "+dpt in text_rdelex):
                            # print(dpt,text_rdelex)
                            flag = False
                            continue

                    for id_tr in all_trainID:
                        if(id_tr in text_rdelex):
                            # print(id_tr,text_rdelex)
                            flag = False
                            continue
                rdelex_conv.append({"spk":conv_turn["spk"],"text":conv_turn["text"],"text_rdelex":text_rdelex})
            elif conv_turn["spk"] in ["SYS-API"]:
                if "trainID-correct" in total_lexer.keys():
                    for ids_v, v in enumerate(total_lexer["trainID-correct"]):
                        text_rdelex = text_rdelex.replace(v,f"[trainID-correct_{ids_v}]")
                        text_rdelex = text_rdelex.replace(v.replace("the ",""),f"[trainID-correct_{ids_v}]")
                for id_tr in all_trainID:
                    if(id_tr in text_rdelex):
                        # print(id_tr,text_rdelex)
                        flag = False
                        continue
                rdelex_conv.append({"spk":conv_turn["spk"],"text":conv_turn["text"],"text_rdelex":text_rdelex})
            else:
                rdelex_conv.append({"spk":conv_turn["spk"],"text":conv_turn["text"],"text_rdelex":text_rdelex})
        if(flag):
            data.append({"id":id_dialogue,"conv":rdelex_conv, "lexer":lexicalized, "dict_original":total_lexer})
        # for conv_turn in rdelex_conv:
        #     print(f"{conv_turn['spk']} >>> {conv_turn['text_rdelex']}")
        # print()
        # print()
        # print()
        # input()

with open('MultiWOZ_2.1/TRAIN_SINGLE_TEMPLATE.json', 'w') as fp:
    json.dump(data, fp, indent=4)

print(good, count_flag, skip1, skip2, skip3, skip4)
print(len(data))
print(good)
print(al)
print(good/float(al))