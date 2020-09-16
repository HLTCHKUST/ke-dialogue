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
        if v == "" or v == "dontcare" or v == 'not mentioned' or v == "don't care" or v == "dont care" or v == "do n't care":
            pass
        else:
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
            all_rows[i].sort()
            time = int(all_rows[i][leaveAt_id])
            mins=int(time%60)
            hours=int(time/60)
            if(len(str(hours)))==1: hours = "0"+str(hours)
            if(len(str(mins)))==1: mins = "0"+str(mins)
            all_rows[i][leaveAt_id] = str(hours)+str(mins)
    if(arriveBy_id!= -1):
        for i in range(len(all_rows)):
            all_rows[i] = list(all_rows[i])
            all_rows[i].sort()
            time = int(all_rows[i][arriveBy_id])
            mins=int(time%60)
            hours=int(time/60)
            if(len(str(hours)))==1: hours = "0"+str(hours)
            if(len(str(mins)))==1: mins = "0"+str(mins)
            all_rows[i][arriveBy_id] = str(hours)+str(mins)
    return all_rows

def get_entity_by_type(info,clmn,post_fix="-info"):
    ### get goal information
    query = to_query("restaurant", info, clmn)
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
    elif(domain in ["hotel","resturant"] and semi['name']!="not mentioned"):
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
            if( (att not in ["none","Time","People","Ticket","Ref","Day"] and val not in ["-","?"]) or (k=="Train-Inform" and att=="Ticket") ):
                if(att in ["Leave","Arrive"]):
                    if( normalize(val).isdecimal() and len(normalize(val))==4):
                        dictionary[att.lower()].append(normalize(val))
                elif(att=="Ticket"):
                    dictionary[att.lower()].append(normalize(val).replace(" each","").replace(" per person",""))
                else:
                    dictionary[att.lower()].append(normalize(val))
                    # if("street" in val.lower()):
                    #     dictionary[att.lower()].append(normalize(val).replace(" street ",""))

    for n in entity_info:
        if(normalize(n) in text_delex):
            dictionary["name"].append(normalize(n))
    # if("name" in info and normalize(info["name"]) in text_delex):
    #     dictionary["name"].append(normalize(info["name"]))
    # for k,v in entity_info.items():
    #     for val in v:
    #         if(type(val)==int and str(val) in text_delex):
    #             dictionary[k].append(str(val))
    #         else:
    #             if(normalize(val) in text_delex):
    #                 dictionary[k].append(normalize(val))
    #             elif("street" in val.lower() and normalize(val).replace(" street","") in text_delex):
    #                 dictionary[k].append(normalize(val).replace(" street",""))
    #                 turns['text'] = turns['text'].replace(normalize(val).replace(" street",""),normalize(val))


    return text_delex


def get_name(conv,dict_delex):
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

all_address = [r[0] for r in database.execute("SELECT DISTINCT address FROM restaurant").fetchall()if r[0] not in ["-","?"]]
all_area = [r[0] for r in database.execute("SELECT DISTINCT area FROM restaurant").fetchall()if r[0] not in ["-","?"]]
all_food = [r[0] for r in database.execute("SELECT DISTINCT food FROM restaurant").fetchall()if r[0] not in ["-","?"]]
all_name = [r[0] for r in database.execute("SELECT DISTINCT name FROM restaurant").fetchall()]
all_phone = [r[0] for r in database.execute("SELECT DISTINCT phone FROM restaurant").fetchall() if r[0] not in ["-","?"]]
all_postcode = [r[0] for r in database.execute("SELECT DISTINCT postcode FROM restaurant").fetchall()if r[0] not in ["-","?"]]
all_pricerange = [r[0] for r in database.execute("SELECT DISTINCT pricerange FROM restaurant").fetchall()if r[0] not in ["-","?"]]

# print(all_address)
# print(all_area)
# print(all_food)
# # print(all_name)
# # print(all_phone)
# # print(all_postcode)
# print(all_pricerange)

dialogue_mwoz = json.load(open("MultiWOZ_2.1/data.json"))
test_split = open("MultiWOZ_2.1/testListFile.txt","r").read()
val_split = open("MultiWOZ_2.1/valListFile.txt","r").read()
train, valid, test = get_splits(dialogue_mwoz,test_split,val_split)
split_by_single_and_domain = json.load(open("dialogue_by_domain.json"))

def generate_all_query(dict_delex,info):
    contrains = [[None],[None],[None]]
    query_template = {}
    results = {}
    mapper_contrains = {0:"area",1:"food",2:"price"}
    if("area" in dict_delex):
        contrains[0] = all_area
        query_template["area"] = ""
        results["area"] = ""
    if("food" in dict_delex):
        contrains[1] = all_food
        query_template["food"] = ""
        results["food"] = ""
    if("price" in dict_delex):
        contrains[2] = all_pricerange
        query_template["price"] = ""
        results["price"] = ""

    results["name"] = ""
    if("addr" in dict_delex):
        results["address"] = ""
    if("phone" in dict_delex):
        results["phone"] = ""
    if("post" in dict_delex):
        results["postcode"] = ""
    lexicalized = []
    if(all([1 if v[0]==None else 0 for v in contrains])):
        for n in all_name:
            query = {"name":n}
            clmn = ["area","pricerange","food","name","address","phone","postcode"]
            database.execute(to_query("restaurant", query, clmn))
            row = database.fetchall() ## a single row
            if len(row) ==1: 
                row = row[0]
            else: 
                continue
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
            clmn = ["area","pricerange","food","name","address","phone","postcode"]
            database.execute(to_query("restaurant", query, clmn).replace("price=","pricerange="))
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

cnt_single_entity_templates = []
al = 0
good = 0
data = []
for k, dial in train.items():
    if(k.lower() in split_by_single_and_domain["restaurant_single"]):
        id_dialogue = k.lower()
        goal = dial["goal"]['restaurant']
        # pp.pprint(goal)
        dictionary = defaultdict(list)
        
        entity_info = get_entity_by_type(goal['info'],["name"],"")
        conversation = []
        train_ID_BOOKED = ""
        state = {"train":0,"attraction":0,"hotel":0,"restaurant":0,"hospital":0,"police":0,"taxi":0,"bus":0}
        for turns in dial["log"]:
            if(turns['metadata']):
                text_delex = delexer(turns,dictionary,entity_info)
                (book, results), state = check_metadata(turns['metadata'],state)
                if(book):
                    entities_by_domain_book = get_booking_query(book)
                    book_delex = book

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
                text_delex =  delexer(turns,dictionary,entity_info)

                conversation.append({"spk":"USR","text":normalize(turns['text']),"text_delex":text_delex,"span_info": turns["span_info"]})
        # for conv_turn in conversation:
        #     print(f"{conv_turn['spk']} >>> {conv_turn['text']}")


        r_delex_dictionary = {}
        len_r_DICT = {}
        for key, d in dictionary.items():
            r_delex_dictionary[key] = list(set(substringSieve(d)))
            len_r_DICT[key] = len(list(set(substringSieve(d))))
            # print(key, list(set(substringSieve(d))))

        al += 1

        cnt_single_entity_templates.append(all([1 if v==1 else 0 for k,v in len_r_DICT.items()]))
        if(not all([1 if v==1 else 0 for k,v in len_r_DICT.items()])):
            continue

        name_resturant, idx_resturant = get_name(conversation,r_delex_dictionary)
        if(name_resturant == None):
            continue

        good += 1
        lexicalized = generate_all_query(r_delex_dictionary, goal['info'])
        # for l in lexicalized:
        #     print(l)


        # pp.pprint(total_lexer)
        # print()
        rdelex_conv = []
        flag = True
        for conv_turn in conversation:
            text_rdelex = conv_turn["text"]
            if conv_turn["spk"] in ["USR","SYS"]:
                if "name" in r_delex_dictionary.keys():
                    for ids_v, v in enumerate(r_delex_dictionary["name"]):
                        text_rdelex = text_rdelex.replace(v,f"[name_{ids_v}]")
                        text_rdelex = text_rdelex.replace(v.replace("the ",""),f"[name_{ids_v}]")
                for ty,val in r_delex_dictionary.items():
                    if(ty != "choice"):
                        for ids_v, v in enumerate(sorted(val, reverse=True, key=lambda item: len(item))):
                            text_rdelex = text_rdelex.replace(v,f"[{ty}_{ids_v}]")

                if "choice" in r_delex_dictionary.keys():
                    numbers = [str(s) for s in text_rdelex.split() if s.isdigit()]
                    for ids_v, v in enumerate(r_delex_dictionary["choice"]):
                        if (v in numbers):
                            text_rdelex = text_rdelex.replace(v,f"[choice_{ids_v}]")

                if("cambridge towninfo centre" not in text_rdelex and "towninfo centre" not in text_rdelex and "cambridge" not in text_rdelex):
                    for food in all_food:
                        if(food in text_rdelex):
                            # print(food,text_rdelex)
                            flag = False
                            continue
                    for phon in all_phone:
                        if(phon in text_rdelex):
                            # print(phon,text_rdelex)
                            flag = False
                            continue
                    for post in all_postcode:
                        if(post in text_rdelex):
                            # print(post,text_rdelex)
                            flag = False
                            continue
                    for price in all_pricerange:
                        if(price in text_rdelex):
                            # print(price,text_rdelex)
                            flag = False
                            continue

                    for add in all_address:
                        if(add in text_rdelex):
                            # print(add,text_rdelex)
                            flag = False
                            continue

                    for nam in all_name:
                        if(nam in text_rdelex):
                            # print(nam,text_rdelex)
                            flag = False
                            continue
                    for area in all_area:
                        if(area in text_rdelex):
                            # print(area,text_rdelex)
                            flag = False
                            continue
                
                rdelex_conv.append({"spk":conv_turn["spk"],"text":conv_turn["text"],"text_rdelex":text_rdelex})
            elif conv_turn["spk"] in ["SYS-API"]:
                if "name" in r_delex_dictionary.keys():
                    for ids_v, v in enumerate(r_delex_dictionary["name"]):
                        text_rdelex = text_rdelex.replace(v,f"[name_{ids_v}]")
                        text_rdelex = text_rdelex.replace(v.replace("the ",""),f"[name_{ids_v}]")

                if("cambridge towninfo centre" not in text_rdelex and "towninfo centre" not in text_rdelex and "cambridge" not in text_rdelex):
                    for nam in all_name:
                        if(nam in text_rdelex):
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
        # print()
        # print()
        # input()
with open('MultiWOZ_2.1/REST_SINGLE_TEMPLATE.json', 'w') as fp:
    json.dump(data, fp, indent=4)

print(len(data))
print(good)
print(al)
print(good/float(al))