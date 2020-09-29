import os,sys
import argparse
import pandas as pd
import timeit
import copy
import json
from utils import filter_columns_from_kb, query_equal_from_kb, query_unequal_from_kb
from revertible_string import RevertibleString

# babi6_filter_keys = ['@R_cuisine', '@R_location', '@R_price']

###
# BABI Function
###

class BABI7Lexicalizer():        
    # Constant definition
    api_call_pattern = 'api_call'
    delex_prefix = '@'
    delex_keys = ['@address', '@area', '@food', '@location', '@phone', '@pricerange', '@postcode', '@type', '@id', '@name']
    filter_keys = ['@food', '@area', '@pricerange']

    # Function to read knwoledge base as pandas dataframe sorted by rating from the given path
    @staticmethod
    def read_knowledge_base(path):
        kb_dict = json.load(open(path, 'r'))
        for i in range(len(kb_dict)):
            kb_item = kb_dict[i]
            for k in kb_item.keys():
                if(k == "postcode"):
                    kb_item[k] = kb_item[k].replace(".","").replace(",","").replace(" ","").lower()
                else: 
                    kb_item[k] = kb_item[k].replace(" ","_").lower()

        kb = pd.DataFrame.from_records(kb_dict).fillna('<UNK>')
        kb.columns = [f'@{column}' for column in kb.columns]
        kb['@food'] = kb['@food'].fillna('international')
        kb['@phone'] = kb['@phone'].fillna('01223_000000')
        return kb

    # Function to read knwoledge base modifier and update the existing kb
    @staticmethod
    def modify_knowledge_base(kb, path):
        raise NotImplementedError
        # return kb

    # Function to read dialogue from the given path
    @staticmethod
    def read_dialogue(template_path):
        dialogues = []
        dialogue = []
        for line in open(template_path,'r').readlines():
            if len(line) == 1: # Only \n
                dialogues.append(dialogue)
                dialogue = []
            else:
                first_space_index = line.index(' ')
                turn_id = line[:first_space_index]
                conv = line[first_space_index+1:].replace('\n','')
                request, response = conv.split('\t')
                dialogue.append((turn_id, RevertibleString(request), RevertibleString(response)))
        return dialogues
    
    # Function to generate metadata from all BABI dialogues
    @staticmethod
    def generate_metadata(dialogues):
        delexicalized_dialog_meta = [] # Buffer containing list of tuple(dialog, delex_dict, delex_resolved_args_list, max_delex_index)
        for dialogue in dialogues:
            delex_to_chat_dict = { } # Dictionary of recoderd delexicalized word to list of Chat object containing the corresponding word
            delex_resolved_args_list = [] # List of all recorded delexicalized words in api_call that need to be resolved for generation
            max_delex_index = 0
            query_max_delex_index = 0
            for turn_id, request, response in dialogue:
                # Process request & response
                for chat in [request, response]:
                    if BABI7Lexicalizer.delex_prefix in chat.str:
                        for delex_key in BABI7Lexicalizer.delex_keys:
                            # TODO : harcoded, max number of entity in babi_7 is only 7, should change this one with count if possible
                            for i in range(1, 9):
                                recorded_delex_word = f'{delex_key}_{i}'
                                if recorded_delex_word in chat.str:
                                    if recorded_delex_word not in delex_to_chat_dict:
                                        delex_to_chat_dict[recorded_delex_word] = []
                                    delex_to_chat_dict[recorded_delex_word].append(chat)
                                    if max_delex_index < i:
                                        max_delex_index = i

                # If api_call
                if response.str.startswith(BABI7Lexicalizer.api_call_pattern):
                    delex_words = response.str.split(' ')[1:]
                    delex_resolved_args = []
                    for delex_word in delex_words:
                        if delex_word.startswith(BABI7Lexicalizer.delex_prefix):
                            delex_resolved_args.append(delex_word)
                            index = int(delex_word[-1])

                    query_max_delex_index = max_delex_index
                    delex_resolved_args_list.append(delex_resolved_args)

            # Add result to global metadata buffer
            delexicalized_dialog_meta.append((dialogue, delex_to_chat_dict, delex_resolved_args_list, max_delex_index, query_max_delex_index))
        return delexicalized_dialog_meta

    # Generate knowledge base index function
    @staticmethod
    def generate_kb_index(kb):
        possible_filter_keys_list = [  # TODO: FU**IN hardcoded combination
            BABI7Lexicalizer.filter_keys, # 3 Keys
            BABI7Lexicalizer.filter_keys[:2], BABI7Lexicalizer.filter_keys[1:], BABI7Lexicalizer.filter_keys[::2], # 2 Keys
            [BABI7Lexicalizer.filter_keys[0]], [BABI7Lexicalizer.filter_keys[1]], [BABI7Lexicalizer.filter_keys[2]] # 1 Key
        ]

        default_index = pd.DataFrame({'index':['_'],'filter_type':['_'],'num_entity':[kb.shape[0]],'kb':[kb]}).set_index('index')
        index_kbs = [default_index]
        for possible_filter_keys in possible_filter_keys_list:
            possible_queries_df = kb[possible_filter_keys].drop_duplicates()
            filter_type = '_'.join(possible_filter_keys)

            index_keys = []
            filter_types = []
            kb_sizes = []
            filtered_kbs = []
            for row in possible_queries_df.to_dict('records'):
                filters = [(attr,value) for attr, value in row.items()]
                filtered_kb = query_equal_from_kb(kb, filters)
                index_keys.append('_'.join([value for value in row.values()]))
                kb_sizes.append(filtered_kb.shape[0])
                filter_types.append(filter_type)
                filtered_kbs.append(filtered_kb) 
                
            index_data = {'index':index_keys,'filter_type':filter_types,'num_entity':kb_sizes,'kb':filtered_kbs}
            index_kbs.append(pd.DataFrame(index_data).set_index('index'))
        index_kb = pd.concat(index_kbs)
        return index_kb

    # Generate dialogue index function
    @staticmethod
    def generate_dialogue_index(dialogues):
        delexicalized_dialog_meta = BABI7Lexicalizer.generate_metadata(dialogues)

        meta_data_list = []
        num_entity_list = []
        filter_type_list = []
        for dialogue_meta in delexicalized_dialog_meta:
            _, _, delex_resolved_args_list, max_delex_index, query_max_delex_index = dialogue_meta
            meta_data_list.append(dialogue_meta)
            num_entity_list.append(max_delex_index - query_max_delex_index)
            if len(delex_resolved_args_list) > 0:
                 # There is only 1 api_call maximum in babi7, process the first element if any
                if len(delex_resolved_args_list[0]) == 0:
                     # There is api_call with no delexicalized parameter
                    filter_type_list.append('_')
                else:
                    filter_type_list.append('_'.join([delex_word[:-2] for delex_word in delex_resolved_args_list[0]]))
            else:
                 # If there is no api_call, add to global index
                filter_type_list.append('_')
        index_dialog = pd.DataFrame({'filter_type':filter_type_list, 'num_entity':num_entity_list,'meta':meta_data_list})
        return index_dialog

    # Generate dialogue by kb and dialogue
    @staticmethod
    def generate_dialogue(filtered_kb, kb, dialogue_meta, random_seed=0):
        # e.g:
        #   1 good morning	hello what can i help you with today
        #   2 can you make a restaurant reservation in @R_location_1 in a @R_price_1 price range	i'm on it
        #   3 <SILENCE>	any preference on a type of cuisine
        #   4 with @R_cuisine_1 food	how many people would be in your party
        #   5 we will be @R_number_1	ok let me look into some options for you
        #   6 <SILENCE>	api_call @R_cuisine_1 @R_location_1 @R_number_1 @R_price_1
        #   7 instead could it be in a @R_price_2 price range	sure is there anything else to update
        #   8 no	ok let me look into some options for you
        #   9 <SILENCE>	api_call @R_cuisine_1 @R_location_1 @R_number_1 @R_price_2
        #   31 <SILENCE>	what do you think of this option: @R_restaurant_3
        #   32 do you have something else	sure let me find an other option for you
        #   33 <SILENCE>	what do you think of this option: @R_restaurant_4
        #   34 no this does not work for me	sure let me find an other option for you
        #   35 <SILENCE>	what do you think of this option: @R_restaurant_5
        #   36 let's do it	great let me do the reservation
        #   37 do you have its phone number	here it is @R_phone_5
        #   38 do you have its address	here it is @R_address_5
        #   39 you rock	is there anything i can help you with
        #   40 no thank you	you're welcome
        #
        # We first resolve the entity on the second API call and then resolve the other 1st entity that has not been resolved
        # For the entities after query, we resolve those entities from the filtered kb with the 2nd API call entities

        # Unfold metadata
        dialogue, delex_to_chat_dict, delex_resolved_args_list, max_delex_index, query_max_delex_index = dialogue_meta

        # There is only 0 or 1 API call for BABI 7, take the first one if any
        if len(delex_resolved_args_list) > 0:
            delex_resolved_args = delex_resolved_args_list[-1] 
        else:
            delex_resolved_args = None

        # Resolved buffer
        resolved_dict = {f'@type_{i}':'restaurant' for i in range(1, max_delex_index+1)}

        # Resolve query on api_call
        ent_max_api_kb = filtered_kb[BABI7Lexicalizer.filter_keys]
        ent_max_api_row = ent_max_api_kb.to_dict('records')[0]

        ent_query_filters = {}
        if delex_resolved_args:
            for delex_resolved_word in delex_resolved_args:
                last_underscore_index = delex_resolved_word.rindex('_')
                delex_key = delex_resolved_word[:last_underscore_index]
                delex_index = int(delex_resolved_word[last_underscore_index+1:])

                resolved_dict[delex_resolved_word] = ent_max_api_row[delex_key]
                ent_query_filters[delex_key] = ent_max_api_row[delex_key]

        # Filter out entity_1 based on the fields on API call and value from entity_2
        for key in BABI7Lexicalizer.filter_keys:
            # Resolve all entities in query number
            if key in ent_query_filters:
                ent_before_possible_values = kb.loc[kb[key] != ent_query_filters[key], :]
            else:
                ent_before_possible_values = kb
            ent_size = ent_before_possible_values.shape[0]
            ent_before_possible_values = ent_before_possible_values[key].sample(ent_size, random_state=random_seed).unique()
            for i in range(1, query_max_delex_index+1):
                if f'{key}_{i}' not in resolved_dict:
                    resolved_dict[f'{key}_{i}'] = ent_before_possible_values[(i-1) % len(ent_before_possible_values)]

        # Filter knowledge for the other entities
        other_ent_kb = filtered_kb.sample(filtered_kb.shape[0], random_state=random_seed)
        num_other_ent = max_delex_index - query_max_delex_index

        # Resolve other entities with top-k entities
        for i in range(num_other_ent):
            for key in BABI7Lexicalizer.delex_keys:
                resolved_dict[f'{key}_{i + query_max_delex_index + 1}'] = other_ent_kb.loc[i, key]

        # Generate new dialogue
        for delex_word, knowledge_value in resolved_dict.items():
            if delex_word in delex_to_chat_dict:
                for chat in delex_to_chat_dict[delex_word]:
                    chat.str = chat.str.replace(delex_word, knowledge_value)

        # Generate string version of the new dialogue
        str_dialogue = []
        for turn_id, request, response in dialogue:
            str_dialogue.append((turn_id, request.str, response.str))

        # Reset all RevertibleString to the original chat
        for delex_word in delex_to_chat_dict.keys():
            for chat in delex_to_chat_dict[delex_word]:
                chat.to_origin()

    #     print('mdi, qmdi', max_delex_index, query_max_delex_index)
    #     print('rd', resolved_dict)
        return str_dialogue

    # Dump generated dialogues to file
    @staticmethod
    def dump_dialogue_to_file(str_dialogues, output_path):
        f = open(output_path,'w')
        for i, str_dialogue in enumerate(str_dialogues):
            for turn_id, request, response in str_dialogue:
                f.write(f'{turn_id} {request}\t{response}\n')
                
            if i != len(str_dialogues) - 1:
                f.write('\n')

if __name__ == '__main__':
    # Sample Command:
    # python generate_dialogues_CAMREST.py --dialogue_path ./CamRest/train_record-delex.txt --knowledge_path ./CamRest/KB.json --output_folder ./output --num_augmented_knowledge 10 --num_augmented_dialogue 10 --random_seed 0

    # Parse args
    parser = argparse.ArgumentParser(description='Generation BABI5')
    parser.add_argument('--dialogue_path', type=str, default='./CamRest/train_record-delex.txt', help='dialog path, default: ./CamRest/train_record-delex.txt')
    parser.add_argument('--knowledge_path', type=str, default='./CamRest/KB.json', help='knowledge base path, default:./CamRest/KB.json')
    parser.add_argument('--knowledge_modifier_path', type=str, help='knowledge base modifier path, default: None')
    parser.add_argument('--output_folder', type=str, default='./', help='output folder path for generation result, default: ./')
    parser.add_argument('--num_augmented_knowledge', type=int, default=10, help='number of augmented knowledge, default: 10')
    parser.add_argument('--num_augmented_dialogue', type=int, default=10, help='number of augmented dialogue, default: 10')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed for reproducible sampling, default: 0')
    args = vars(parser.parse_args())

    # Print begin information
    print('== Selective Generation BABI 7 Dialogue ==')
    print(args)
    
    # Start Timer
    start = timeit.default_timer()  
    
    # Args
    dialogue_path = args["dialogue_path"]
    knowledge_path = args["knowledge_path"]
    knowledge_modifier_path = args["knowledge_modifier_path"]
    
    num_augmented_knowledge = args['num_augmented_knowledge']
    num_augmented_dialogue = args['num_augmented_dialogue']
    random_seed = args['random_seed']
    
    output_folder = args['output_folder']
    output_path = f'{output_folder}/gen-babi7-nk{num_augmented_knowledge}-nd{num_augmented_dialogue}-rs{random_seed}.txt'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Read KB & dataset
    kb = BABI7Lexicalizer.read_knowledge_base(knowledge_path)
    if knowledge_modifier_path:
        kb = BABI7Lexicalizer.modify_knowledge_base(kb, knowledge_modifier_path, BABI7Lexicalizer.filter_keys)
    dialogues = BABI7Lexicalizer.read_dialogue(dialogue_path)

    # Generate kb and dialog index
    index_kb = BABI7Lexicalizer.generate_kb_index(kb)
    index_dialog = BABI7Lexicalizer.generate_dialogue_index(dialogues)

    # Sampled KB
    sampled_kb = index_kb.sample(num_augmented_knowledge, random_state=random_seed)
    str_dialogues = []
    for i in range(num_augmented_knowledge):
        filtered_kb = sampled_kb['kb'][i]
        filtered_kb_size = sampled_kb['num_entity'][i]
        filter_type = sampled_kb['filter_type'][i]

        # Retrieve valid dialogue meta
        row_filters = (index_dialog['filter_type'] == filter_type) & (index_dialog['num_entity'] <= filtered_kb_size)
        dialogue_metas = index_dialog.loc[row_filters, 'meta']
        if dialogue_metas.shape[0] >= num_augmented_dialogue:
            num_possible_dialogue_augmentation = num_augmented_dialogue
        else:
            num_possible_dialogue_augmentation = dialogue_metas.shape[0]
        dialogue_metas = dialogue_metas.sample(num_possible_dialogue_augmentation, random_state=random_seed+i)

        # Generate possible (dialogue_meta, filtered KB) pairs to maximize the knowledge coverage
        filtered_kb = filtered_kb.sample(filtered_kb.shape[0], random_state=random_seed)
        num_dialogue_generated = 0
        kb_idx = 0
        dialog_kb_pairs = []
        for dialogue_meta in dialogue_metas:
            max_delex_index, query_max_delex_index = dialogue_meta[3], dialogue_meta[4]
            num_entity = max_delex_index - query_max_delex_index
            if kb_idx + num_entity >= filtered_kb.shape[0]:
                num_rest_entity = num_entity - (filtered_kb.shape[0] - kb_idx)
                kb_segment = pd.concat([filtered_kb.iloc[kb_idx:,:], filtered_kb.iloc[:num_rest_entity,:]]).reset_index(drop=True)
                dialog_kb_pairs.append((dialogue_meta, kb_segment))
                kb_idx = num_rest_entity
            else:
                kb_segment = filtered_kb.iloc[kb_idx : kb_idx + num_entity,:].reset_index(drop=True)
                dialog_kb_pairs.append((dialogue_meta, kb_segment))
                kb_idx = kb_idx + num_entity

        # Generate Dialogue
        for pair_idx, (dialogue_meta, kb_segment) in enumerate(dialog_kb_pairs):
            str_dialogue = BABI7Lexicalizer.generate_dialogue(kb_segment, kb, dialogue_meta, random_seed=random_seed +i+pair_idx)
            str_dialogues.append(str_dialogue)

    BABI7Lexicalizer.dump_dialogue_to_file(str_dialogues, output_path)
    
    # Print Execution time
    stop = timeit.default_timer()

    # Print end information
    print(f'File saved on `{output_path}`')
    print(f'Execution time: {stop - start:.4}s') 
