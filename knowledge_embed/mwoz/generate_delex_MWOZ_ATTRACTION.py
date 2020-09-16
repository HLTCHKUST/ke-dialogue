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

timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat = re.compile("\d{1,3}[.]\d{1,2}")

def set_seed(val):
    np.random.seed(val)
    
set_seed(42)

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
    # if isinstance(text, int) or isinstance(text, float):
    #     return text

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
        # if v == "swimmingpool": v = "swimming pool"
        # if v == "nightclub": v = "night club"
        # if v == "the golden curry": v = "golden curry"
        # if v == "mutliple sports": v = "multiple sports"
        # if v == "the cambridge chop house": v = "cambridge chop house"
        # if v == "the fitzwilliam museum": v = "fitzwilliam museum"
        # if v == "the good luck chinese food takeaway": v = "good luck chinese food takeaway"
        # if v == "the cherry hinton village centre": v = "cherry hinton village centre"
        # if v == "the copper kettle": v = "copper kettle"
        # if v == "pizza express Fen Ditton": v = "pizza express"
        # if v == "shiraz restaurant": v = "shiraz"
        # # if v == "christ's college": v = "christ college"
        # if v == "good luck chinese food takeaway": v = "chinese"

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
            time = all_rows[i][leaveAt_id]
            mins=int(time%60)
            hours=int(time/60)
            if(len(str(hours)))==1: hours = "0"+str(hours)
            if(len(str(mins)))==1: mins = "0"+str(mins)
            all_rows[i][leaveAt_id] = str(hours)+str(mins)
    if(arriveBy_id!= -1):
        for i in range(len(all_rows)):
            all_rows[i] = list(all_rows[i])
            time = all_rows[i][arriveBy_id]
            mins=int(time%60)
            hours=int(time/60)
            if(len(str(hours)))==1: hours = "0"+str(hours)
            if(len(str(mins)))==1: mins = "0"+str(mins)
            all_rows[i][arriveBy_id] = str(hours)+str(mins)
    return all_rows

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

def get_booking_query(text):
    domain = {"global":set(),"train":[],"attraction":[],"hotel":[],"restaurant":[],"taxi":[],
              "police":[],"hospital":[],"generic":[]}
    domain[text.split()[0]] = re.findall(r"'(.*?)'", text)
    return domain

def get_name(conv,dict_delex):
    for conv_turn in reversed(conv):
        if "name" in dict_delex.keys():
            for ids_v, v in enumerate(r_delex_dictionary["name"]):
                if(v in conv_turn["text"]):
                    return v, ids_v
                if(v.replace("the ","") in conv_turn["text"]):
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












###########################
# FUNCTIONS
###########################
def check_metadata(dic, state):
    for d, v in dic.items():
        if (state[d]==0 or state[d]!= v['book']['booked']):
            if (len(v['book']['booked']) > 0):
                state[d] = v['book']['booked']
                return parse_results(v['book'],v['semi'],d), state 
            for k, v1 in v['book'].items():
                if(k != 'booked' and v1 != ""):
                    return parse_results(v['book'],v['semi'],d), state
    return (None, None), state

# was called delex
def delexer(turns,dictionary,entity_info):
    text_delex = normalize(turns['text'])
    
    for k, v in turns['dialog_act'].items():
        for [act, val] in v: # key: dialog act, val: entitie(s)
            if (act not in ["none"] and val not in ["-", "?"]):
                if(act=="Choice" and val == "79"):
                    pass
                else:
                    dictionary[act.lower()].append(normalize(val))    
    for k,v in entity_info.items():
        if(v not in ["-", "?"]  and normalize(v) in text_delex):

            dictionary[k].append(normalize(v))

    return text_delex

def get_entity_by_type(info,clmn,post_fix="-info"):
    ### get goal information
    query = to_query("attraction", info, clmn)
    database.execute(query)
    all_rows = database.fetchall()
    all_rows = convert_time_int_to_time(all_rows,clmn)
    entity_by_type = {c+post_fix:set() for c in clmn}
    for rows in all_rows:
        for i,c in enumerate(clmn):
            entity_by_type[c+post_fix].add(rows[i])
    # entity_by_type["number_of_options"] = [len(all_rows)]
    return entity_by_type

def get_attraction_name(conv, dict_delex):
    """
    Get attraction name
    """
    for conv_turn in reversed(conv):
        if "name" in dict_delex.keys():
            for ids_v, v in enumerate(r_delex_dictionary["name"]):
                if(v in conv_turn["text"]):
                    return v, ids_v
                if(v in conv_turn["text"]):
                    return v, ids_v
    return None, None

# 0|location||0||0
# 1|pricerange||0||0
# 2|id||0||0
# 3|entrance|fee|0||0
# 4|type||0||0
# 5|openhours||0||0
# 6|phone||0||0
# 7|area||0||0
# 8|postcode||0||0
# 9|name||0||0
# 10|address||0||0

pp = pprint.PrettyPrinter(indent=4)
conn = sqlite3.connect('MWOZ.db')
database = conn.cursor()
all_area = [r[0] for r in database.execute("SELECT DISTINCT area FROM attraction").fetchall()]
all_name = [r[0] for r in database.execute("SELECT DISTINCT name FROM attraction").fetchall()]
all_type = [r[0] for r in database.execute("SELECT DISTINCT type FROM attraction").fetchall()]

def generate_all_query(dict_delex,info):
    contrains = [[None],[None]]
    query_template = {}
    results = {}
    mapper_contrains = {0:"area",1:"type"}
    if("area" in dict_delex):
        contrains[0] = all_area
        query_template["area"] = ""
        results["area"] = ""
    if("type" in dict_delex):
        contrains[1] = all_type
        query_template["type"] = ""
        results["type"] = ""
    results["name"] = ""
    if("phone" in dict_delex):
        results["phone"] = ""
    if("addr" in dict_delex):
        results["address"] = ""
    if("post" in dict_delex):
        results["postcode"] = ""
    if("fee" in dict_delex):
        results["entrance fee"] = ""
    lexicalized = []
    if(all([1 if v[0]==None else 0 for v in contrains])):
        for n in all_name:
            query = {"name":n}
            clmn = ["address","area","name","phone","postcode","entrance fee","type"]
            database.execute(to_query("attraction", query, clmn))
            row = database.fetchall()[0] ## a single row
            if("entrance fee" in results and row[clmn.index('entrance fee')] == "?"):
                continue
            choice = "1"
            if(row[2]!= dict_delex["name"]):
                result = results.copy()
                for k in results.keys():
                    result[k] = row[clmn.index(k)]
                if("choice" in dict_delex):
                    result["choice"] = choice
                lexicalized.append(result)
    else:
        for combo in product(*contrains):
            query = query_template.copy()

            for c_i, c in enumerate(combo):
                if(c!= None):
                    query[mapper_contrains[c_i]] = c
            clmn = ["address","area","name","phone","postcode","entrance fee","type"]
            database.execute(to_query("attraction", query, clmn))
            all_rows = database.fetchall()
            choice = str(len(all_rows))
            if(len(all_rows)>0):
                for row in all_rows:
                    if("entrance fee" in results and row[clmn.index('entrance fee')] == "?"):
                        continue
                    if(row[2]!= dict_delex["name"]):
                        result = results.copy()
                        for k in results.keys():
                            result[k] = row[clmn.index(k)]
                        if("choice" in dict_delex):
                            result["choice"] = choice
                        lexicalized.append(result)
    return lexicalized

# read db
dialogue_mwoz = json.load(open("MultiWOZ_2.1/data.json"))
test_split = open("MultiWOZ_2.1/testListFile.txt","r").read()
val_split = open("MultiWOZ_2.1/valListFile.txt","r").read()
train, valid, test = get_splits(dialogue_mwoz,test_split,val_split)
split_by_single_and_domain = json.load(open("dialogue_by_domain.json"))
al = 0
good = 0
# clmn_train = ["id",'location','departure','destination','leaveAt']
cnt_single_entity_templates = []
data = []
for k, dial in train.items():
    if(k.lower() in split_by_single_and_domain["attraction_single"]):
        id_dialogue = k.lower()
        goal = dial["goal"]["attraction"]
        # pp.pprint(goal)

        # take all the names
        dictionary = defaultdict(list)
        if("name" in goal['info']): 
            # dictionary["id"].append(normalize(goal['info']["id"]))
            entity_info = get_entity_by_type(goal['info'],["address","area","name","phone","postcode","entrance fee","type"],"")
            entity_info = {k.replace("entrance fee","fee").replace("address","addr"): list(v)[0].lower() for k,v in entity_info.items()}
            # print(entity_info)
        else:
            entity_info = {}
            # print(entity_info)

        # get all entities
        # pp.pprint(entity_info)

        conversation = []
        state = {"train":0, "attraction":0, "hotel":0, "restaurant":0, "hospital":0, "police":0, "taxi":0, "bus":0}
        for turns in dial["log"]:
            if (turns['metadata']):
                text_delex = delexer(turns, dictionary, entity_info)
                (_, results), state = check_metadata(turns['metadata'], state) # skip booking
                conversation.append({"spk":"SYS","text":normalize(turns['text']),"text_delex":text_delex,"span_info": turns["span_info"]})
            else:
                text_delex = delexer(turns, dictionary, entity_info)
                conversation.append({"spk":"USR","text":normalize(turns['text']),"text_delex":text_delex,"span_info": turns["span_info"]})

        # print the conversation
        # for conv_turn in conversation:
        #     print(f"{conv_turn['spk']} >>> {conv_turn['text']}")

        # merge substring into one
        r_delex_dictionary = {}
        len_r_dict = {} # count
        for key, d in dictionary.items():
            r_delex_dictionary[key] = list(set(substringSieve(d)))
            len_r_dict[key] = len(list(set(substringSieve(d))))
            # print(key, list(set(substringSieve(d))))
        al += 1

        if(not all([1 if v==1 else 0 for k,v in len_r_dict.items()])):
            continue

        name_attraction, idx_attraction = get_name(conversation,r_delex_dictionary)
        if(name_attraction == None):
            continue
        
        lexicalized = generate_all_query(r_delex_dictionary, goal['info'])
        # for l in lexicalized:
        #     print(l)
        good += 1
        flag = True
        rdelex_conv = []
        for conv_turn in conversation:
            text_rdelex = conv_turn["text"]
            if conv_turn["spk"] in ["USR","SYS"]:

                if "name" in r_delex_dictionary.keys():
                    for ids_v, v in enumerate(r_delex_dictionary["name"]):
                        text_rdelex = text_rdelex.replace(v,f"[name_{ids_v}]")
                        text_rdelex = text_rdelex.replace(v.replace("the ",""),f"[name_{ids_v}]")
                if "type" in r_delex_dictionary.keys():
                    for ids_v, v in enumerate(r_delex_dictionary["type"]):
                        if(v == "nightclubs"): v = "nightclub"
                        if(v == "museums"): v = "museum"
                        if(v == "cinemas"): v = "cinema"
                        if(v == "kings hedges learner pools"): v = "kings hedges learner pool"
                        
                        text_rdelex = text_rdelex.replace(v,f"[type_{ids_v}]")
                        text_rdelex = text_rdelex.replace(v.replace("the ",""),f"[type_{ids_v}]")
                if "choice" in r_delex_dictionary.keys():
                    numbers = [str(s) for s in text_rdelex.split() if s.isdigit()]
                    for ids_v, v in enumerate(r_delex_dictionary["choice"]):
                        if (v in numbers):
                            text_rdelex = text_rdelex.replace(v,f"[choice_{ids_v}]")

                for ty,val in r_delex_dictionary.items():
                    if(ty not in ["name","choice"]):
                        for ids_v, v in enumerate(sorted(val, reverse=True, key=lambda item: len(item))):
                            text_rdelex = text_rdelex.replace(v,f"[{ty}_{ids_v}]")
    
                # all_name
                if("cambridge towninfo centre" not in text_rdelex and "towninfo centre" not in text_rdelex and "cambridge" not in text_rdelex):
                    for area in all_area:
                        if(" "+area in text_rdelex):
                            flag = False
                            continue

                    for nam in all_name:
                        if(nam in text_rdelex):
                            flag = False
                            continue

                    if "entries for attractions . are you looking for architecture , colleges , parks , nightclubs , or maybe a museum ?" not in text_rdelex:
                        for typ in all_type:
                            if(typ in text_rdelex):
                                flag = False
                                continue
                            if(typ.replace("mutliple","") in text_rdelex):
                                flag = False
                                continue
                             
                rdelex_conv.append({"spk":conv_turn["spk"],"text":conv_turn["text"],"text_rdelex":text_rdelex})
            elif conv_turn["spk"] in ["SYS-API"]:
                if "name" in r_delex_dictionary.keys():
                    for ids_v, v in enumerate(r_delex_dictionary["name"]):
                        text_rdelex = text_rdelex.replace(v,f"[name_{ids_v}]")
                        text_rdelex = text_rdelex.replace(v.replace("the ",""),f"[name_{ids_v}]")
                rdelex_conv.append({"spk":conv_turn["spk"],"text":conv_turn["text"],"text_rdelex":text_rdelex})
            else:
                rdelex_conv.append({"spk":conv_turn["spk"],"text":conv_turn["text"],"text_rdelex":text_rdelex})
        if(flag):
            data.append({"id":id_dialogue,"conv":rdelex_conv, "lexer":lexicalized, "dict_original":r_delex_dictionary})
        # for conv_turn in rdelex_conv:
        #     print(f"{conv_turn['spk']} >>> {conv_turn['text_rdelex']}")
        # # input()
        # print()
        # print()
        # print()
with open('MultiWOZ_2.1/ATTR_SINGLE_TEMPLATE.json', 'w') as fp:
    json.dump(data, fp, indent=4)
print(len(data))
print(good)
print(al)
print(good/float(al))