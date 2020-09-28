from transformers import (AdamW,WEIGHTS_NAME, CONFIG_NAME)
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from utils.hugging_face import top_filtering, build_input_from_segments
from utils.hugging_face import get_loader, load_model, get_parser, SPECIAL_TOKENS, MODEL_INPUTS, add_special_tokens_, average_distributed_scalar, make_logdir, add_token_bAbI
from utils.preprocessSMD import load_SMD, generate_dataset_FINETUNE

import os
import os.path
import math
import random
import numpy as np
import warnings
import json
import jsonlines
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser
from pprint import pformat
import torch
import torch.nn.functional as F

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
        if(args.unilm):
            att_mask = instance["attention_mask"].unsqueeze(0).unsqueeze(0).to(input_ids.device)
            if(args.graph or args.edge_list):
                att_mask = att_mask.squeeze().squeeze()
                max_l = len(instance["input_ids"]) + len(instance["input_graph_ids"])
                max_r = len(instance["input_graph_ids"])
                mask_padded = torch.zeros(max_l, max_l, dtype=torch.long,device=args.device)
                mask_padded[max_r:len(att_mask[0])+max_r,max_r:len(att_mask[0])+max_r].copy_(att_mask)
                ## add missing one for row
                row_stripe_padded = torch.ones(max_r, max_r+instance["len_token_a"]+1, dtype=torch.long, device=args.device)
                mask_padded[:max_r,:max_r+instance["len_token_a"]+1].copy_(row_stripe_padded)
                ## add missing one for clmn
                cmn_stripe_padded = torch.ones(len(att_mask[0]), max_r, dtype=torch.long, device=args.device)
                mask_padded[max_r:max_r+len(att_mask[0]),:max_r].copy_(cmn_stripe_padded)
                if(args.adj_graph):
                    r_net = len(instance["input_graph_networks"]) ## square matrix 
                    c_net = len(instance["input_graph_networks"][0]) ## square matrix 
                    if(r_net and c_net):
                        mask_padded[:r_net,:r_net].copy_(torch.tensor(instance["input_graph_networks"],dtype=torch.long, device=args.device))
                att_mask = mask_padded.unsqueeze(0).unsqueeze(0)

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


def finetune_model(args,model,loader):
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    def update(engine, batch):
        model.train()
        batch = tuple(batch[input_name].to(args.device) for input_name in MODEL_INPUTS)
        input_ids, lm_labels, token_type_ids, nodes_ids, attention_mask = batch
        if(not args.graph and not args.edge_list): 
            nodes_ids = None
        if(not args.unilm): attention_mask = None  
        (lm_loss), *_ = model(
            input_ids=input_ids, token_type_ids=token_type_ids, labels=lm_labels, 
            nodes=nodes_ids,attention_mask=attention_mask
        )
        loss = lm_loss / args.gradient_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(batch[input_name].to(args.device) for input_name in MODEL_INPUTS)
            input_ids, lm_labels, token_type_ids, nodes_ids, attention_mask = batch
            if(not args.graph and not args.edge_list): 
                nodes_ids = None
            if(not args.unilm): attention_mask = None  
            # if we dont send labels to model, it doesnt return losses
            lm_logits, *_ = model(input_ids=input_ids, token_type_ids=token_type_ids, nodes=nodes_ids, 
                                  attention_mask=attention_mask)
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted,), (lm_labels_flat_shifted,)

    trainer = Engine(update)
    evaluator = Engine(inference)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(loader))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

        # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(), output_transform=lambda x: (x[0][0], x[1][0]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=["loss"])
    evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))
    trainer.run(loader, max_epochs=args.n_epochs)
    return model

def get_training_file_for_KB(args,indx,tokenizer):
    train_kb = generate_dataset_FINETUNE(f"data/SMD/test/dialog_{indx}.txt",tokenizer)
    data = {"train":train_kb,"valid":train_kb, "test":train_kb}
    train_loader, _, _ = get_loader(args, data, tokenizer)
    return train_loader

if __name__ == "__main__":
    args = get_parser()
    
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # Get model and tokenizer
    model, tokenizer = load_model(args,load=True)

    print("Load Data")
    if(args.dataset == "SMD"):
        test, global_ent = load_SMD(args, tokenizer, test_flag=True)
    else: 
        print("ERROR: select a dataset with --dataset [SMD|MWOZ|DIALKG]")
        exit(1)

    # dataset split
    inds = args.eval_indices.split(",")
    start_ind, end_ind = 0, len(test)
    if len(inds) == 2:
        start_ind, end_ind = int(inds[0]), int(inds[1])

    j_output = defaultdict(list)
    filter_domains = args.filter_domain.split(",")
    for i, conv in tqdm(enumerate(test),total=len(test)):
        if i < start_ind or i > end_ind:
            continue

        # skip the domain in filter domains
        if(len(conv["edges"]) > 0) and (conv['domain'] not in filter_domains):
            ## load KB
            train_KB_loader = get_training_file_for_KB(args,i,tokenizer)
            ## finetune
            args.n_epochs = 30

            model = finetune_model(args,model,train_KB_loader) 
        else:
            print(f"{i} skip {conv['domain']} domain!")
        for sample in conv['dialogue']:  
            out_ids = sample_sequence(sample['history'],sample["graph"] if args.dataset == "DIALKG" else conv,tokenizer, model, args) 
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            j_output[conv["id"]].append({"spk":sample['spk'],"text":out_text})
        ## reload original models
        if(len(conv["edges"]) > 0):
            model, tokenizer = load_model(args,load=True)
    with open(args.model_checkpoint+'/result_splits/'+ str(start_ind) + '_' + str(end_ind) + '.json', 'w') as fp:
        json.dump(j_output, fp, indent=4)
