import os,sys
import argparse
import pandas as pd
import timeit
import copy
from utils import filter_columns_from_kb, query_equal_from_kb, query_unequal_from_kb
from revertible_string import RevertibleString

# babi6_filter_keys = ['@R_cuisine', '@R_location', '@R_price']

###
# BABI Function
###

class BABI5Lexicalizer():        
    # Constant definition
    api_call_pattern = 'api_call'
    delex_prefix = '@R_'
    delex_keys = ['@R_cuisine', '@R_restaurant', '@R_location', '@R_price', '@R_phone', '@R_address', '@R_number']
    filter_keys = ['@R_cuisine', '@R_location', '@R_price', '@R_number']

    # Function to read knwoeldge base as dictionary from the given path
    @staticmethod
    def get_type_dict_flatten(kb_path, dstc2=False): 
        type_dict = {'R_restaurant':[]}
        fd = open(kb_path,'r') 

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

            # Add Restaurant per address
            if entity == 'R_address':
                type_dict['R_restaurant'].append(rest_name)

            # Add Entity Value
            if entity not in type_dict.keys():
                type_dict[entity] = []
            type_dict[entity].append(entity_value)
        return type_dict

    # Function to read knwoledge base as pandas dataframe sorted by rating from the given path
    @staticmethod
    def read_knowledge_base(path):
        kb_dict = BABI5Lexicalizer.get_type_dict_flatten(path)
        kb = pd.DataFrame(kb_dict)
        kb.columns = [f'@{column}' for column in kb.columns]
        kb.sort_values('@R_rating', ascending=False, inplace=True)
        return kb

    # Function to read knwoledge base modifier and update the existing kb
    @staticmethod
    def modify_knowledge_base(kb, path, filter_keys):
        mod_kb_dict = BABI5Lexicalizer.get_type_dict_flatten(path)
        mod_kb = pd.DataFrame(mod_kb_dict)
        mod_kb.columns = [f'@{column}' for column in mod_kb.columns]

        # Remove duplicated keys from old tuples
        remove_keys_df = mod_kb[filter_keys].drop_duplicates().set_index(filter_keys)
        kb = kb.set_index(filter_keys)
        kb = kb[~kb.index.isin(remove_keys_df.index)].reset_index()

        # Add new items from KB modifier & resort by rating
        kb = pd.concat([kb[kb.columns], mod_kb[kb.columns]])
        kb.sort_values('@R_rating', ascending=False, inplace=True)

        return kb
        
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
            for turn_id, request, response in dialogue:
                if response.str.startswith(BABI5Lexicalizer.api_call_pattern):
                    delex_words = response.str.split(' ')[1:]
                    delex_resolved_args_list.append(delex_words)

                for chat in [request, response]:
                    if BABI5Lexicalizer.delex_prefix in chat.str:
                        for delex_key in BABI5Lexicalizer.delex_keys:
                            # TODO : harcoded, max number of entity in babi_5 is only 8, should change this one with count if possible
                            for i in range(1, 9):
                                recorded_delex_word = f'{delex_key}_{i}'
                                if recorded_delex_word in chat.str:
                                    if recorded_delex_word not in delex_to_chat_dict:
                                        delex_to_chat_dict[recorded_delex_word] = []
                                    delex_to_chat_dict[recorded_delex_word].append(chat)

                                    if max_delex_index < i:
                                        max_delex_index = i

            # Add result to global metadata buffer
            delexicalized_dialog_meta.append((dialogue, delex_to_chat_dict, delex_resolved_args_list, max_delex_index))
        return delexicalized_dialog_meta

    # Generate knowledge base index function
    @staticmethod
    def generate_kb_index(kb):
        possible_queries_df = kb[BABI5Lexicalizer.filter_keys].drop_duplicates()

        index_keys = []
        kb_sizes = []
        filtered_kbs = []
        for row in possible_queries_df.to_dict('records'):
            filters = [(attr,value) for attr, value in row.items()]
            filtered_kb = query_equal_from_kb(kb, filters)

            index_keys.append('_'.join([value for value in row.values()]))
            filtered_kbs.append(filtered_kb)
            kb_sizes.append(filtered_kb.shape[0])
        index_kb = pd.DataFrame({'index':index_keys,'num_entity':kb_sizes,'kb':filtered_kbs}).set_index('index')

        return index_kb

    # Generate dialogue index function
    @staticmethod
    def generate_dialogue_index(dialogues):
        delexicalized_dialog_meta = BABI5Lexicalizer.generate_metadata(dialogues)

        meta_data_list = []
        num_entity_list = []
        for dialogue, delex_to_chat_dict, delex_resolved_args_list, max_delex_index in delexicalized_dialog_meta:
            meta_data_list.append((dialogue, delex_to_chat_dict, delex_resolved_args_list, max_delex_index))
            num_entity_list.append(max_delex_index - 2)
        index_dialog = pd.DataFrame({'num_entity':num_entity_list,'meta':meta_data_list})

        return index_dialog

    # Generate dialogue by kb and dialogue
    @staticmethod
    def generate_dialogue(filtered_kb, kb, dialogue_meta):
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

        # Unfold meta dialogue
        dialogue, delex_to_chat_dict, delex_resolved_args_list, max_delex_index = dialogue_meta
        
        # There is only 1 valid API call for BABI 5, which is the last one
        delex_resolved_args = delex_resolved_args_list[-1] 

        # Resolved buffer
        resolved_dict = {}

        # Resolve entity_2 from query
        ent_2_kb = filtered_kb[BABI5Lexicalizer.filter_keys]
        ent_2_row = ent_2_kb.to_dict('records')[0]
        for key in ent_2_row.keys():
            resolved_dict[f'{key}_2'] = ent_2_row[key]

        # Create filter for entity_1
        ent_1_filters = []
        for delex_resolved_word in delex_resolved_args:
            last_underscore_index = delex_resolved_word.rindex('_')
            delex_key = delex_resolved_word[:last_underscore_index]
            delex_index = delex_resolved_word[last_underscore_index+1:]

            if delex_index == '1':
                ent_1_filters.append((delex_resolved_word[:-2], resolved_dict[f'{delex_key}_2']))

        # Filter out entity_1 based on the fields on API call and value from entity_2
        ent_1_kb = query_unequal_from_kb(kb, ent_1_filters)
        ent_1_row = ent_1_kb.to_dict('records')[0]

        # Resolve entity_1
        for key in ent_1_row.keys():
            if f'{key}_1' in delex_resolved_args:
                resolved_dict[f'{key}_1'] = resolved_dict[f'{key}_2']
            else:
                resolved_dict[f'{key}_1'] = ent_1_row[key]

        # Filter knowledge for the other entities
        other_ent_kb = filtered_kb
        num_other_ent = max_delex_index - 2

        # Resolve other entities with top-k entities sorted by rating
        # The rating is already sorted when reading the file, no need to do sorting here
        for i in range(num_other_ent):
            for key in BABI5Lexicalizer.delex_keys:
                resolved_dict[f'{key}_{i + 3}'] = other_ent_kb.loc[i, key]

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
    # python generate_dialogues_babi5.py --dialogue_path ./dialog-bAbI-tasks/dialog-babi-task5trn_record-delex.txt --knowledge_path ./dialog-bAbI-tasks/dialog-babi-kb-all.txt --output_folder ./output --num_augmented_knowledge 10 --num_augmented_dialogue 10 --random_seed 0
    
    # Parse args
    parser = argparse.ArgumentParser(description='Generation BABI5')
    parser.add_argument('--dialogue_path', type=str, default='./dialog-bAbI-tasks/dialog-babi-task5trn_record-delex.txt', help='dialog path, default: ./dialog-bAbI-tasks/dialog-babi-task5trn_record-delex.txt')
    parser.add_argument('--knowledge_path', type=str, default='./dialog-bAbI-tasks/dialog-babi-kb-all.txt', help='knowledge base path, default: ./dialog-bAbI-tasks/dialog-babi-kb-all.txt')
    parser.add_argument('--knowledge_modifier_path', type=str, default='./dialog-bAbI-tasks/dialog-babi-kb-oov-modifier.txt', help='knowledge base modifier path, default: ./dialog-babi-kb-oov-modifier.txt')
    parser.add_argument('--output_folder', type=str, default='./', help='output folder path for generation result, default: ./')
    parser.add_argument('--num_augmented_knowledge', type=int, default=10, help='number of augmented knowledge, default: 10')
    parser.add_argument('--num_augmented_dialogue', type=int, default=10, help='number of augmented dialogue, default: 10')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed for reproducible sampling, default: 0')
    args = vars(parser.parse_args())
    
    # Print begin information
    print('== Selective Generation BABI 5 Dialogue ==')
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
    output_path = f'{output_folder}/gen-babi5-nk{num_augmented_knowledge}-nd{num_augmented_dialogue}-rs{random_seed}.txt'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Read KB & dataset
    kb = BABI5Lexicalizer.read_knowledge_base(knowledge_path)
    if knowledge_modifier_path:
        kb = BABI5Lexicalizer.modify_knowledge_base(kb, knowledge_modifier_path, BABI5Lexicalizer.filter_keys)
    dialogues = BABI5Lexicalizer.read_dialogue(dialogue_path)

    # Generate kb and dialog index
    index_kb = BABI5Lexicalizer.generate_kb_index(kb)
    index_dialog = BABI5Lexicalizer.generate_dialogue_index(dialogues)
    
    # Generate dialogues  
    sampled_kb = index_kb.sample(num_augmented_knowledge, random_state=random_seed)
    str_dialogues = []
    for i in range(num_augmented_knowledge):
        row_index = sampled_kb.index[i]
        filtered_kb = sampled_kb['kb'][i]
        filtered_kb_size = sampled_kb['num_entity'][i]

        dialogue_metas = index_dialog.loc[index_dialog['num_entity'] <= filtered_kb_size, 'meta'].sample(num_augmented_dialogue, random_state=random_seed)
        for dialogue_meta in dialogue_metas:
            str_dialogue = BABI5Lexicalizer.generate_dialogue(filtered_kb, kb, dialogue_meta)
            str_dialogues.append(str_dialogue)
            
    # Dump dialogues to file
    BABI5Lexicalizer.dump_dialogue_to_file(str_dialogues, output_path)
    
    # Print Execution time
    stop = timeit.default_timer()

    # Print end information
    print(f'File saved on `{output_path}`')
    print(f'Execution time: {stop - start:.4}s') 
