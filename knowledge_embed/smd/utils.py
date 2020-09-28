# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from datetime import datetime
import os
import socket
from itertools import chain
import math
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import torch.nn.functional as F
from transformers import (AdamW, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, ReformerLMHeadModel, ReformerConfig, DIALOGGPT2LMHeadModel,WEIGHTS_NAME, CONFIG_NAME)
from transformers import BertTokenizer, BertForMaskedLM, SimpleTokenizer
import json

SPECIAL_TOKENS = ["<bos>", "<|endoftext|>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': ' <|endoftext|>', 'pad_token': '<pad>',
                         'additional_special_tokens': ('<speaker1>', '<speaker2>')}


MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids","input_graph_ids","attention_mask"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
PADDED_SPECIAL = ["input_graph_ids", "input_graph_networks"]

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer) #len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def build_input_from_segments(args, history, reply, graph, tokenizer, lm_labels=False, with_eos=True, valid="train"):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """

    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    if(args.flatten_KB and with_eos): history += graph['edges']
    history = history[-args.max_history:]
    sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["lm_labels"] = [-100] * len(instance["input_ids"])

    instance["input_graph_ids"] = None
    instance["input_graph_networks"] = None
    if args.edge_list:
        instance["input_graph_ids"] = graph['edges']
    elif(args.graph):
        instance["input_graph_ids"] = graph['nodes']
        if(args.adj_graph):
            instance["input_graph_networks"] = graph['adj_mat']

    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

    ### attetion mask
    instance["attention_mask"] = None
    instance["len_token_a"] = 0
    instance["padding_token"] = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
    if(args.unilm):
        len_seq = len(instance["lm_labels"])    
        input_mask = torch.zeros(len_seq, len_seq, dtype=torch.long)
        tril_matrix = torch.tril(torch.ones((len_seq, len_seq), dtype=torch.long))
        len_token_a = len(list(chain(*history))) + len(history) + 1 # +1 BOS and len(history) account for the special token in between the turns   
        len_token_b = len(list(chain(*[reply + ([eos] if with_eos else [])]))) + 1
        assert len_token_a + len_token_b == len(instance["lm_labels"])
        input_mask[:len_token_a + len_token_b+1, :len_token_a+1].fill_(1)
        second_st, second_end = len_token_a, len_token_a+len_token_b
        input_mask[second_st:second_end, second_st:second_end].copy_(tril_matrix[:second_end-second_st, :second_end-second_st])
        # from numpy import save
        # save('unilm.npy', input_mask.cpu().numpy())
        instance["attention_mask"] = input_mask
        instance["len_token_a"] = len_token_a
    return instance


def get_loader(args,data_raw, tokenizer):
    # if not os.path.isfile("data/MultiWOZ_2.1/train_segment.p"):
    print("Building Sequences")
    # datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    datasets = {"train": [], "valid": [], "test": []}
    # domains, particularly for MWOZ
    domains = {"train": [], "valid": [], "test": []} 
    
    for dataset_name, data in data_raw.items():
        for conv in tqdm(data,total=len(data)):
            for sample in conv['dialogue']: 
                instance = build_input_from_segments(args,sample['history'], sample["response"], sample["graph"] if args.dataset == "DIALKG" else conv, tokenizer, lm_labels=True, with_eos=True, valid=dataset_name)   
                # for input_name, input_array in instance.items():
                datasets[dataset_name].append(instance)
                if args.dataset == "MWOZ" or args.dataset == "MWOZ_SINGLE":
                    domains[dataset_name].append(conv['domain'])

    print("Build train and validation dataloaders")
    train_dataset = DatasetTrain(datasets["train"], domains["train"])
    valid_dataset = DatasetTrain(datasets["valid"], domains["valid"])
    test_dataset = DatasetTrain(datasets["test"], domains["test"])

    if args.balance_sampler:
        print("setup imbalanced sampler")
        train_loader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size=args.train_batch_size, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,collate_fn=collate_fn)
    
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False,collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.valid_batch_size, shuffle=False,collate_fn=collate_fn)

    print("Train: {}".format(len(train_dataset)))
    print("Valid: {}".format(len(valid_loader)))
    print("Test: {}".format(len(test_loader)))
    return train_loader, valid_loader, test_loader


def test_dataloader(args,data_load):
    max_len = 0
    for batch in data_load:
        if(batch['input_ids'].shape[1]> max_len): 
            max_len = batch['input_ids'].shape[1]
        if(batch['input_ids'].shape[1] > args.max_seq_len):
            print(f"Input seq is too long: {batch['input_ids'].shape[1]}")
            exit(0)
    return max_len



class DatasetTrain(Dataset):
    """Custom data.Dataset compatible with DataLoader."""
    def __init__(self, data, domains=None):
        self.data = data
        self.dataset_len = len(self.data)
        self.domains = domains

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = self.data[index]
        return item
        
    def __len__(self):
        return self.dataset_len

def collate_fn(data):
    padding = data[0]["padding_token"]
    ## just load node and adj if needed 
    net = True if(data[0]["input_graph_ids"]) else False
    adj_net = True if(data[0]["input_graph_networks"]) else False

    max_l = max(len(x["input_ids"]) for x in data)
    padded_dataset = {n:[] for n in MODEL_INPUTS}
    for x in data:
        padded_dataset["token_type_ids"].append( x["token_type_ids"]+ [padding]*(max_l-len(x["input_ids"])) )
        padded_dataset["lm_labels"].append( x["lm_labels"]+ [-100]*(max_l-len(x["lm_labels"]))  )
        padded_dataset["input_ids"].append(x["input_ids"]+ [padding]*(max_l-len(x["input_ids"])))

    ## nodes embedding
    if(net):
        max_r = max(len(mat["input_graph_ids"]) for mat in data)
        max_c = max(len(col) for mat in data for col in mat["input_graph_ids"])
        matrixes = []
        for mat in data:
            temp = []
            for clmn in mat["input_graph_ids"]:
                temp.append(clmn + [padding] * (max_c - len(clmn)))
            temp += [[padding] * max_c] * (max_r - len(mat["input_graph_ids"]))
            matrixes.append(temp)
        padded_dataset["input_graph_ids"] = matrixes
        max_l += max_r 

    ## attention mask
    if(data[0]["len_token_a"]>0):
        attention_mat_padded = []
        for i, att_mask in enumerate(data):
            mask_padded = torch.zeros(max_l, max_l, dtype=torch.long)
            if(net):
                mask_padded[max_r:len(att_mask["attention_mask"][0])+max_r,max_r:len(att_mask["attention_mask"][0])+max_r].copy_(att_mask["attention_mask"])
                ## add missing one for row
                row_stripe_padded = torch.ones(max_r, max_r+data[i]["len_token_a"]+1, dtype=torch.long)
                mask_padded[:max_r,:max_r+data[i]["len_token_a"]+1].copy_(row_stripe_padded)
                ## add missing one for clmn
                cmn_stripe_padded = torch.ones(len(att_mask["attention_mask"][0]), max_r, dtype=torch.long)
                mask_padded[max_r:max_r+len(att_mask["attention_mask"][0]),:max_r].copy_(cmn_stripe_padded)
                if(adj_net):
                    r_net = len(data[i]["input_graph_networks"]) ## square matrix 
                    c_net = len(data[i]["input_graph_networks"][0]) ## square matrix 
                    if(r_net and c_net):
                        mask_padded[:r_net,:r_net].copy_(torch.tensor(data[i]["input_graph_networks"],dtype=torch.long))
            else:
                mask_padded[:len(att_mask["attention_mask"][0]),:len(att_mask["attention_mask"][0])].copy_(att_mask["attention_mask"])

            # from numpy import save
            mask_padded = mask_padded.unsqueeze(0)
            attention_mat_padded.append(mask_padded.tolist())

        padded_dataset["attention_mask"] = attention_mat_padded

    for input_name in MODEL_INPUTS:
        padded_dataset[input_name] = torch.tensor(padded_dataset[input_name])
    return padded_dataset

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def make_logdir(args,model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    # current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # logdir = os.path.join('runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    logdir = os.path.join('runs', f'{args.dataset}_{model_name}_graph_{args.graph}_adj_{args.adj_graph}_edge_{args.edge_list}_unilm_{args.unilm}_flattenKB_{args.flatten_KB}_historyL_{args.max_history}_lr_{args.lr}_epoch_{args.n_epochs}_weighttie_{args.weight_tie}_kbpercentage_{args.kbpercentage}_layer_{args.layers}_balancesampler_{args.balance_sampler}_upsampler_{args.up_sampler}') #  current_time + '_' + socket.gethostname() + '_' + model_name)
    
    return logdir


def load_modelDGPT(model, checkpoint, args, verbose=False):
    if checkpoint is None or checkpoint == "None":
        if verbose:
            print('No checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            print('Loading finetuned model from %s' % checkpoint)
        model_state_dict = torch.load(checkpoint)

        model_state_dict = fix_state_dict_namespace(model_state_dict)

        start_model = model
        if (hasattr(model, "transformer")
            and all(not s.startswith('transformer.')
                    for s in model_state_dict.keys())):
            print('Loading transfomer only')
            start_model = model.transformer
        start_model.load_state_dict(model_state_dict)

    return model


def fix_state_dict_namespace(model_state_dict):
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t.startswith('module.'):
            new_key = t.replace('module.', '')
        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    return model_state_dict

def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

def load_model(args, load=False):
    if 'gpt2-bAbI' in args.model_checkpoint:
        if(load):
            tokenizer = SimpleTokenizer(args.model_checkpoint,load=True)
            model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
        else:
            if("BABI" in args.dataset): tokenizer = SimpleTokenizer(get_bAbI_vocab())
            if("DSTC" in args.dataset): tokenizer = SimpleTokenizer(get_DSTC_vocab())
            if("CAMREST" in args.dataset): tokenizer = SimpleTokenizer(get_CAMREST_vocab())
            config = GPT2Config()
            config.n_layer = args.layers
            config.vocab_size = len(tokenizer)
            model = GPT2LMHeadModel(config)
        args.max_seq_len = 1024
    elif 'gpt2_scratch_tiny' in args.model_checkpoint:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        config = GPT2Config()
        config.n_layer = args.layers
        if(load):
            model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
        else:
            model = GPT2LMHeadModel(config)
        args.max_seq_len = 1024
    elif 'reformer_gpt2' in args.model_checkpoint:
        config = ReformerConfig()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        pretrained_gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        if(load):
            model = ReformerLMHeadModel.from_pretrained(args.model_checkpoint)
        else:
            model = ReformerLMHeadModel(config)
            model.load_pretrained_gpt2_weights(config, pretrained_gpt2_model)
        del pretrained_gpt2_model
    elif 'gpt2' in args.model_checkpoint:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
        model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
        args.max_seq_len = 1024
    elif 'reformer_scratch' in args.model_checkpoint or 'reformer_scratch_tiny' in args.model_checkpoint :
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        config = ReformerConfig()
        if 'reformer_scratch_tiny' in args.model_checkpoint:
            config.n_layer = args.layers
        config.weight_tie = args.weight_tie
        model = ReformerLMHeadModel(config)
    elif 'bert' in args.model_checkpoint:
        if(load):
            tokenizer = BertTokenizer.from_pretrained(args.model_checkpoint)
            model = BertForMaskedLM.from_pretrained(args.model_checkpoint)
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            model = BertForMaskedLM.from_pretrained('bert-base-cased')
        model.config.is_decoder = True
        args.max_seq_len = 512
    elif 'DGPT' in args.model_checkpoint:
        args.model_size = 'small'
        args.model_path = f'data/dialoGPT/{args.model_size}/'
        config = GPT2Config.from_json_file(os.path.join(args.model_path, 'config.json'))
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        model = load_modelDGPT(DIALOGGPT2LMHeadModel(config), args.model_path+f"{args.model_size}_ft.pkl", args, verbose=True)
        args.max_seq_len = 1024

    model.to(args.device)
    if 'gpt2-bAbI' not in args.model_checkpoint:
        add_special_tokens_(model,tokenizer)
    if 'DGPT' in args.model_checkpoint:
        model.set_tied()
    return model,tokenizer

def get_bAbI_vocab():
    def generate_dataset(data_split):
        num_lines = sum(1 for line in open(data_split,'r'))
        with open(data_split,'r') as f:
            conversation = []
            data = []
            for line in f:
                if(line == "\n"):
                    data.append(conversation)
                    conversation = []
                else:
                    _, line = line.replace("\n","").split(' ', 1)
                    if ("\t" in line):
                        user, syst = line.split("\t")
                        if("<SILENCE>" not in user):
                            conversation.append({"spk":"USR","text":user})
                        if("i'm on it" not in syst and "api_call" not in syst and "ok let me look into some options for you" not in syst):
                            conversation.append({"spk":"SYS","text":syst})
        return data
    
    entity_file = open("data/dialog-bAbI-tasks/dialog-babi-kb-all.txt")
    vocab = set()
    for line in entity_file: 
        _, line = line.replace("\n","").split(' ', 1)
        s,r,o = line.split()
        vocab.add(s)
        vocab.add(o)
    # print(len(vocab))

    data  = generate_dataset('data/dialog-bAbI-tasks/dialog-babi-task5trn.txt')
    data += generate_dataset('data/dialog-bAbI-tasks/dialog-babi-task5dev.txt')
    data += generate_dataset('data/dialog-bAbI-tasks/dialog-babi-task5tst.txt')
    data += generate_dataset('data/dialog-bAbI-tasks/dialog-babi-task5tst-OOV.txt')
    for conv in data:
        for turns in conv:
            for token in turns['text'].split():
                vocab.add(token)
    vocab = list(vocab)
    vocab.sort()
    vocab = {tok:int(idx) for idx, tok in enumerate(vocab)}
    return vocab

def get_DSTC_vocab():
    def generate_dataset(data_split):
        num_lines = sum(1 for line in open(data_split,'r'))
        with open(data_split,'r') as f:
            conversation = []
            data = []
            for line in f:
                if(line == "\n"):
                    data.append(conversation)
                    conversation = []
                else:
                    _, line = line.replace("\n","").split(' ', 1)
                    if ("\t" in line):
                        user, syst = line.split("\t")
                        if("<SILENCE>" not in user):
                            conversation.append({"spk":"USR","text":user})
                        if("i'm on it" not in syst and "api_call" not in syst and "ok let me look into some options for you" not in syst):
                            conversation.append({"spk":"SYS","text":syst})
        return data
    
    entity_file = open("data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-kb.txt")
    vocab = set()
    for line in entity_file: 
        _, line = line.replace("\n","").split(' ', 1)
        s,_,o = line.split()
        vocab.add(s)
        vocab.add(o)
 

    data  = generate_dataset('data/dialog-bAbI-tasks/dialog-babi-task6trn.txt')
    data += generate_dataset('data/dialog-bAbI-tasks/dialog-babi-task6dev.txt')
    data += generate_dataset('data/dialog-bAbI-tasks/dialog-babi-task6tst.txt')

    for conv in data:
        for turns in conv:
            for token in turns['text'].split():
                vocab.add(token)

    vocab = {tok:int(idx) for idx, tok in enumerate(vocab)}
    return vocab


def get_CAMREST_vocab():
    def generate_dataset(data_split):
        num_lines = sum(1 for line in open(data_split,'r'))
        with open(data_split,'r') as f:
            conversation = []
            data = []
            for line in f:
                if(line == "\n"):
                    data.append(conversation)
                    conversation = []
                else:
                    _, line = line.replace("\n","").split(' ', 1)
                    if ("\t" in line):
                        user, syst = line.split("\t")
                        if("<SILENCE>" not in user):
                            conversation.append({"spk":"USR","text":user})
                        if("i'm on it" not in syst and "api_call" not in syst and "ok let me look into some options for you" not in syst):
                            conversation.append({"spk":"SYS","text":syst})
        return data
    

    KB = json.load(open('data/CamRest/KB.json'))
    vocab = set()
    for item in KB:
        for k, v in item.items():
            if(k =="postcode"):
                vocab.add(v.replace(".","").replace(",","").replace(" ","").lower())   
            else: 
                vocab.add(v.replace(" ","_").lower()) 

    data  = generate_dataset('data/CamRest/train.txt')
    data += generate_dataset('data/CamRest/dev.txt')
    data += generate_dataset('data/CamRest/test.txt')

    for conv in data:
        for turns in conv:
            for token in turns['text'].split():
                vocab.add(token)
    
    vocab.add("<UNK>")
    vocab.add("cheaply")


    vocab = {tok:int(idx) for idx, tok in enumerate(vocab)}
    return vocab

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.domains[idx]
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=1000000000, help="Number of previous exchanges to keep in history")
    parser.add_argument("--max_seq_len", type=int, default=200, help="Max number of tokens")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--graph", action='store_true', help="Use graph in input")
    parser.add_argument("--adj_graph", action='store_true', help="Use graph in input")
    parser.add_argument("--edge_list", action='store_true', help="Use list of edges as input")
    parser.add_argument("--debug", action='store_true', help="debugging mode")
    parser.add_argument("--dataset", type=str, default='SMD', help="Choose between SMD, MWOZ, DIALKG, TASKMASTER, CAMREST")
    parser.add_argument("--unilm", action='store_true', help="UniLM style Seq2Seq")
    parser.add_argument("--flatten_KB", action='store_true', help="flatten gold kb in input")
    parser.add_argument("--max_length", type=int, default=150, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=1, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--weight_tie", action='store_true', help="Use weight tie")
    parser.add_argument("--kbpercentage", type=int, default=0, help="data-aug")
    parser.add_argument("--layers", type=int, default=12, help="data-aug")
    parser.add_argument("--eval_indices", type=str, default="0,100", help="Evaluate indices from i to j (inclusive). Example 0,100")
    parser.add_argument("--filter_domain", type=str, default="")
    parser.add_argument("--balance_sampler", action='store_true', help="Add a balance sampler")
    parser.add_argument("--up_sampler", action='store_true', help="Upsample data for some domains")

    args = parser.parse_args()
    print_opts(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Make sure cuda is deterministic
    torch.backends.cudnn.deterministic = True

    return args