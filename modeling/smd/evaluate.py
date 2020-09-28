from utils.preprocessSMD import load_SMD
from transformers import (AdamW,WEIGHTS_NAME, CONFIG_NAME)
from utils.hugging_face import load_model,get_parser,top_filtering, SPECIAL_TOKENS, add_special_tokens_, average_distributed_scalar, make_logdir, build_input_from_segments,add_token_bAbI
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import os
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
import math
from pprint import pformat
import random
from utils.eval_metrics import moses_multi_bleu, compute_prf, compute_prf_SMD
import numpy as np
from tqdm import tqdm
import warnings
import json
import jsonlines
import os.path
from collections import defaultdict


def sample_sequence(history, graph,tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    padding = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
    if current_output is None:
        current_output = []
    if(args.flatten_KB):  
        history += graph['edges']
    for i in range(args.max_length):
        instance = build_input_from_segments(args,history,current_output,graph,tokenizer, with_eos=False)
        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        nodes_ids = None
        if (args.graph or args.edge_list) and len(instance["input_graph_ids"])>0:
            max_c = max(len(col) for col in instance["input_graph_ids"])
            temp = []
            for clmn in instance["input_graph_ids"]:
                temp.append(clmn + [padding] * (max_c - len(clmn)))
            nodes_ids = torch.tensor([temp], device=args.device)

        att_mask = None

        logits = model(input_ids, token_type_ids=token_type_ids, nodes=nodes_ids, attention_mask=att_mask)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


if __name__ == "__main__":
    args = get_parser()
    
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # Get model and tokenizer
    model, tokenizer = load_model(args,load=True)

    print("Load Data")
    test, _ = load_SMD(args, tokenizer, test_flag=True)

    j_output = defaultdict(list)
    for i, conv in tqdm(enumerate(test),total=len(test)):
        for sample in conv['dialogue']:  
            out_ids = sample_sequence(sample['history'],sample["graph"] if args.dataset == "DIALKG" else conv,tokenizer, model, args) 
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            j_output[conv["id"]].append({"spk":sample['spk'],"text":out_text})

    with open(args.model_checkpoint+'/result.json', 'w') as fp:
        json.dump(j_output, fp, indent=4)

            

        
