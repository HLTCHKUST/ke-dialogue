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
            q += f' {k}>"{v}" and'
        elif k == 'arriveBy':
            q += f' {k}<"{v}" and'
        else:
            q += f' {k}="{v}" and'

    q = q[:-3] ## this just to remove the last AND from the query 
    return q

def get_entity_by_type(info,clmn_hotel):
    ### get goal information
    query = to_query("hotel", info, clmn_hotel)
    database.execute(query)
    all_rows = database.fetchall()
    entity_by_type = {c:set() for c in clmn_hotel}
    for rows in all_rows:
        for i,c in enumerate(clmn_hotel):
            if(c in ["stars","phone"]):
                entity_by_type[c].add(int(rows[i]))
            else:
                entity_by_type[c].add(rows[i])
    entity_by_type["number_of_options"] = [len(all_rows)]
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
            book_query += f" trainID = '{dic_data['booked'][0]['trainID']}'"
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

def delexer(turns,dictionary,entity_name_info,info):
    text_delex = normalize(turns['text'])
    ### first serch using SPEECH ACT
    for k,v in turns['dialog_act'].items():
        # if("Inform" in k or "Select" in k):
        for [att,val] in v:
            if(att == 'Addr'):
                if val.lower() in 'back lane, cambourne':
                    dictionary[att.lower()].append(normalize('back lane, cambourne'))
                elif val.lower() in 'kingfisher way, hinchinbrook business park, huntingdon':
                    dictionary[att.lower()].append(normalize('kingfisher way, hinchinbrook business park, huntingdon'))
                elif val.lower() in '15-17 norman way, coldhams business park':
                    dictionary[att.lower()].append(normalize('15-17 norman way, coldhams business park'))
                elif val.lower() in 'sleeperz hotel, station road':
                    dictionary[att.lower()].append(normalize('sleeperz hotel, station road'))
                else:
                    if(val not in ["-","?","none"]):
                        dictionary[att.lower()].append(normalize(val))

            elif(att not in ['Day','Stay','People','Parking',"Ref","Internet"]):
                if(val not in ["-","?","none"]):
                    dictionary[att.lower()].append(normalize(val))
                    if("the" in val.lower()):
                        dictionary[att.lower()].append(normalize(val).replace("the ",""))
    if(turns['metadata'] and turns['metadata']["hotel"]["semi"]['name'] != "not mentioned"):
        dictionary["name"].append(normalize(turns['metadata']["hotel"]["semi"]['name']))

    
    for n in entity_name_info:
        if(normalize(n) in text_delex):
            dictionary["name"].append(normalize(n))
    if("name" in info and normalize(info["name"]) in text_delex):
        dictionary["name"].append(normalize(info["name"]))


    return text_delex

def get_name_hotel(conv,dict_delex):
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


pp = pprint.PrettyPrinter(indent=4)
conn = sqlite3.connect('MWOZ.db')
database = conn.cursor()
all_areas = [r[0] for r in database.execute("SELECT DISTINCT area FROM hotel").fetchall()]
all_name = [r[0] for r in database.execute("SELECT DISTINCT name FROM hotel").fetchall()]
all_pricerange = [r[0] for r in database.execute("SELECT DISTINCT pricerange FROM hotel").fetchall()]
all_stars = [r[0] for r in database.execute("SELECT DISTINCT stars FROM hotel").fetchall()]
all_type = [r[0] for r in database.execute("SELECT DISTINCT type FROM hotel").fetchall()]
all_address = [r[0] for r in database.execute("SELECT DISTINCT address FROM hotel").fetchall()]
all_postcode = [r[0] for r in database.execute("SELECT DISTINCT postcode FROM hotel").fetchall()]


def generate_all_query(dict_delex,info):
    contrains = [[None],[None],[None],[None]]
    query_template = {}
    results = {}
    mapper_contrains = {0:"area",1:"stars",2:"price",3:"type"}
    if("area" in dict_delex):
        contrains[0] = all_areas
        query_template["area"] = ""
        results["area"] = ""
    if("stars" in dict_delex):
        contrains[1] = all_stars
        query_template["stars"] = ""
        results["stars"] = ""
    if("price" in dict_delex):
        contrains[2] = all_pricerange
        query_template["price"] = ""
        results["price"] = ""
    if("type" in dict_delex):
        contrains[3] = all_type
        query_template["type"] = ""
        results["type"] = ""
    if("parking" in info):
        query_template["parking"] = info["parking"]
    if("internet" in info):
        query_template["internet"] = info["internet"]

    results["name"] = ""
    if("phone" in dict_delex):
        results["phone"] = ""
    if("addr" in dict_delex):
        results["address"] = ""
    if("post" in dict_delex):
        results["postcode"] = ""
    lexicalized = []
    if(all([1 if v[0]==None else 0 for v in contrains])):
        for n in all_name:
            query = {"name":n}
            clmn = ["address","area","name","phone","postcode","pricerange","stars","type"]
            database.execute(to_query("hotel", query, clmn))
            row = database.fetchall()[0] ## a single row
            choice = "1"
            if(row[2]!= dict_delex["name"]):
                result = results.copy()
                for k in results.keys():
                    result[k] = row[clmn.index(k.replace('price','pricerange'))]
                if("choice" in dict_delex):
                    result["choice"] = choice
                lexicalized.append(result)
    else:
        for combo in product(*contrains):
            query = query_template.copy()

            for c_i, c in enumerate(combo):
                if(c!= None):
                    query[mapper_contrains[c_i]] = c
            clmn = ["address","area","name","phone","postcode","pricerange","stars","type"]
            database.execute(to_query("hotel", query, clmn).replace("price=","pricerange="))
            all_rows = database.fetchall()
            choice = str(len(all_rows))
            if(len(all_rows)>0):
                for row in all_rows:
                    if(row[2]!= dict_delex["name"]):
                        result = results.copy()
                        for k in results.keys():
                            result[k] = row[clmn.index(k.replace('price','pricerange'))]
                        if("choice" in dict_delex):
                            result["choice"] = choice
                        lexicalized.append(result)
    return lexicalized





# print(all_areas)
# print(all_name)
# print(all_pricerange)
# print(all_stars)
# print(all_type)
# print(all_address)

dialogue_mwoz = json.load(open("MultiWOZ_2.1/data.json"))
test_split = open("MultiWOZ_2.1/testListFile.txt","r").read()
val_split = open("MultiWOZ_2.1/valListFile.txt","r").read()
train, valid, test = get_splits(dialogue_mwoz,test_split,val_split)
split_by_single_and_domain = json.load(open("dialogue_by_domain.json"))
clmn_hotel = ["name"]
cnt_single_entity_templates = []
data = []
for k, dial in train.items():
    if(k.lower() in split_by_single_and_domain["hotel_single"]):
        id_dialogue = k.lower()
        goal = dial["goal"]['hotel']
        # pp.pprint(goal)
        if 'hotel' in goal:
            continue
        dictionary = defaultdict(list)
        
        if("name" in goal['info']): 
            dictionary["name"].append(normalize(goal['info']["name"]))
            
        entity_info = get_entity_by_type(goal['info'],clmn_hotel)
        
        conversation = []
        state = {"train":0,"attraction":0,"hotel":0,"restaurant":0,"hospital":0,"police":0,"taxi":0,"bus":0}
        for turns in dial["log"]:
            if(turns['metadata']):
                text_delex = delexer(turns,dictionary,entity_info["name"],goal["info"])

                (book, results), state = check_metadata(turns['metadata'],state)
                if(book):
                    entities_by_domain_book = get_booking_query(book)
                    book_delex = book

                    ## delex book 
                    for n in entity_info["name"]:
                        if(normalize(n) in book_delex):
                            dictionary["name"].append(normalize(n))
                    if "name" in dictionary.keys():
                        for v in sorted(dictionary["name"],reverse=True, key=lambda item: len(item)):
                            book_delex = book_delex.replace(v,f"[name]")


                    conversation.append({"entities":entities_by_domain_book,"spk":"SYS-API","text":book,"text_delex":book_delex})
                    dom_API = book.split()[0] ## first token is the API domain
                    ## THIS IS A SIMULATION OF AN API RESULTS
                    if("dialog_act" in turns and dom_API == "train" and "Train-OfferBooked" in turns["dialog_act"]):
                        for elem_ in turns["dialog_act"]["Train-OfferBooked"]:
                            if(elem_[0]=="Ticket" and elem_[1] != "None"):
                                results = str(results)
                                results += " "+ str(elem_[1])
                    # print(results)
                    conversation.append({"spk":"API","text":str(results).lower(),"text_delex":str(results).lower()})

                conversation.append({"spk":"SYS","text":normalize(turns['text']),"text_delex":text_delex,"span_info": turns["span_info"]})
            else:
                text_delex =  delexer(turns,dictionary,entity_info["name"],goal["info"])
                conversation.append({"spk":"USR","text":normalize(turns['text']),"text_delex":text_delex,"span_info": turns["span_info"]})

        # for conv_turn in conversation:
        #     print(f"{conv_turn['spk']} >>> {conv_turn['text_delex']}")

        if("pricerange" in goal['info']): 
            goal['info']["price"] = goal['info'].pop('pricerange')
        r_delex_dictionary = {}
        len_r_DICT = {}
        for key, d in dictionary.items():
            r_delex_dictionary[key] = list(set(substringSieve(d)))
            len_r_DICT[key] = len(list(set(substringSieve(d))))
            # print(key, list(set(substringSieve(d))))
        
        cnt_single_entity_templates.append(all([1 if v==1 else 0 for k,v in len_r_DICT.items()]))
        if(not all([1 if v==1 else 0 for k,v in len_r_DICT.items()])):
            continue
        name_hotel, idx_hotel = get_name_hotel(conversation,r_delex_dictionary)
        if(name_hotel == None):
            continue

        lexicalized = generate_all_query(r_delex_dictionary, goal['info'])
        flag = True
        rdelex_conv = []
        for conv_turn in conversation:
            text_rdelex = conv_turn["text"]
            if conv_turn["spk"] in ["USR","SYS"]:
                if "stars" in r_delex_dictionary.keys():
                    dict_replace = get_start_end_ACT(conv_turn["span_info"])
                    for k,v in dict_replace.items():
                        if str(k) not in r_delex_dictionary["stars"]: 
                            continue
                        ids_v = r_delex_dictionary["stars"].index(str(k))
                        text_rdelex = text_rdelex.split()
                        text_rdelex[v[0]] = f"[stars_{ids_v}]"
                        text_rdelex = " ".join(text_rdelex)
                if "name" in r_delex_dictionary.keys():
                    for ids_v, v in enumerate(r_delex_dictionary["name"]):
                        text_rdelex = text_rdelex.replace(v,f"[name_{ids_v}]")
                        text_rdelex = text_rdelex.replace(v.replace("the ",""),f"[name_{ids_v}]")
                if "choice" in r_delex_dictionary.keys():
                    numbers = [str(s) for s in text_rdelex.split() if s.isdigit()]
                    for ids_v, v in enumerate(r_delex_dictionary["choice"]):
                        if (v in numbers):
                            text_rdelex = text_rdelex.replace(v,f"[choice_{ids_v}]")

                for ty,val in r_delex_dictionary.items():
                    if(ty not in ["stars","name","choice"]):
                        for ids_v, v in enumerate(sorted(val, reverse=True, key=lambda item: len(item))):
                            text_rdelex = text_rdelex.replace(v,f"[{ty}_{ids_v}]")
                if(len([s for s in text_rdelex.split() if s.isdigit()])>0 and "[stars_" in text_rdelex):
                    flag = False
                    continue
                
                for price_r in all_pricerange:
                    if(price_r in text_rdelex):
                        flag = False
                        continue
                for addrs in all_address:
                    if(addrs in text_rdelex):
                        flag = False
                        continue
                if("cambridge towninfo centre" not in text_rdelex and "towninfo centre" not in text_rdelex and "cambridge" not in text_rdelex):
                    for names_h in all_name:
                        if(names_h in text_rdelex):
                            flag = False
                            continue
                        # sng0841.json
                        if(names_h.replace("guest house","") in text_rdelex):
                            flag = False
                            continue
                    for area in all_areas:
                        if(area in text_rdelex):
                            flag = False
                            continue
                
                rdelex_conv.append({"spk":conv_turn["spk"],"text":conv_turn["text"],"text_rdelex":text_rdelex})
            elif conv_turn["spk"] in ["SYS-API"]:
                if "name" in r_delex_dictionary.keys():
                    for ids_v, v in enumerate(r_delex_dictionary["name"]):
                        text_rdelex = text_rdelex.replace(v,f"[name_{ids_v}]")
                        text_rdelex = text_rdelex.replace(v.replace("the ",""),f"[name_{ids_v}]")
                    for names_h in all_name:
                        if(names_h in text_rdelex):
                            flag = False
                            continue
                        # sng0841.json
                        if(names_h.replace("guest house","") in text_rdelex):
                            flag = False
                            continue
                rdelex_conv.append({"spk":conv_turn["spk"],"text":conv_turn["text"],"text_rdelex":text_rdelex})
            else:
                rdelex_conv.append({"spk":conv_turn["spk"],"text":conv_turn["text"],"text_rdelex":text_rdelex})

        if(flag):
            data.append({"id":id_dialogue,"conv":rdelex_conv, "lexer":lexicalized, "dict_original":r_delex_dictionary})
        # for conv_turn in rdelex_conv:
        #     print(f"{conv_turn['spk']} >>> {conv_turn['text_rdelex']}")
        # print()
        # input()

with open('MultiWOZ_2.1/HOTEL_SINGLE_TEMPLATE.json', 'w') as fp:
    json.dump(data, fp, indent=4)

print(len(data))
print(sum(cnt_single_entity_templates))
print(len(cnt_single_entity_templates))
print(sum(cnt_single_entity_templates)/float(len(cnt_single_entity_templates)))
