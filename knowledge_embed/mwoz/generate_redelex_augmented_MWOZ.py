import json
from copy import deepcopy

files = ["MultiWOZ_2.1/ATTR_SINGLE_TEMPLATE.json", "MultiWOZ_2.1/HOTEL_SINGLE_TEMPLATE.json", "MultiWOZ_2.1/REST_SINGLE_TEMPLATE.json", "MultiWOZ_2.1/TRAIN_SINGLE_TEMPLATE.json"]
domains = ["attraction", "hotel", "restaurant", "train"]

def map_attr(attr):
    replacement_map = {"postcode":"post", "entrance fee": "fee", "address": "addr"}

    for key in replacement_map:
        attr = attr.replace(key, replacement_map[key])
    return attr

for num_template in [25, 50, 100]:
    for i in range(len(files)):
        f, domain = files[i], domains[i]

        conversations = {}
        with open(f) as json_file:
            data = json.load(json_file)

            count = 0
            for sample in data:
                count += 1 
                if count > num_template and domain != "restaurant":
                    break
                if count > num_template * 2 and domain == "restaurant":
                    break

                _id = sample["id"]
                turns = sample["conv"]
                lexer = sample["lexer"]
                dict_original = sample["dict_original"]

                for opt in lexer:
                    new_turns = deepcopy(turns)
                    for attr in opt:
                        if isinstance(opt[attr], list):
                            opt[attr] = opt[attr][0]

                        for i in range(len(new_turns)):
                            turn = new_turns[i]
                            if "text_original" not in turn:
                                turn["text_original"] = turn["text"]
                                turn["text"] = turn["text_rdelex"]
                            
                            turn["text"] = turn["text"].replace("[" + map_attr(attr) + "_0]", opt[attr])
                            new_turns[i] = turn
                    for i in range(len(new_turns)):
                        turn = new_turns[i]
                        if "[" in turn or "]" in turn:
                            print("ERROR", turn)
                    new_sample = {
                        "src": _id,
                        "conversation": new_turns,
                        "template": opt
                    }
                    conversations[len(conversations)] = new_sample

        path = f"MultiWOZ_2.1/train/{domain}_augmented_{num_template}_single.json"
        print(domain, num_template, len(data), len(conversations))
        with open(path, "w") as outfile:
            json.dump(conversations, outfile, indent=4)
