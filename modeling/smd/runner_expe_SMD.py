from joblib import Parallel, delayed
from tqdm import tqdm
import queue
import os
import time
import json
import sys

test_samples = {
    "SMD": 304
}

# Define number of GPUs available
GPU_available = [0,0,0]
N_GPU = len(GPU_available)
N_SAMPLES = test_samples["SMD"] # SMD

### generate experiments
N_SAMPLES_PER_PROCESS = N_SAMPLES // N_GPU
experiments = []
split_list = []
model_checkpoint = "runs/SMD_gpt2_graph_False_adj_False_edge_False_unilm_False_flattenKB_False_historyL_1000000000_lr_6.25e-05_epoch_10_weighttie_False_kbpercentage_0_layer_12"

for i in range(N_GPU):
    start_ind, end_ind = i * N_SAMPLES_PER_PROCESS, (i+1) * N_SAMPLES_PER_PROCESS # the range is inclusive
    if i == N_GPU-1: end_ind = N_SAMPLES-1
    command = "python evaluate_finetune.py --dataset SMD --model_checkpoint {} --top_k 1 --eval_indices {},{} --filter_domain {}".format(model_checkpoint, start_ind, end_ind, sys.argv[1])
    experiments.append(command)
    split_list.append((start_ind, end_ind))

# Put indices in queue
q = queue.Queue(maxsize=N_GPU)
mapper = {}
invert_mapper = {}
for i in range(N_GPU):
    mapper[i] = GPU_available[i]
    invert_mapper[GPU_available[i]] = i
    q.put(i)

def runner(cmd):
    gpu = mapper[q.get()]
    print("> running: CUDA_VISIBLE_DEVICES=%d %s" % (gpu, cmd))
    os.system("CUDA_VISIBLE_DEVICES=%d %s" % (gpu, cmd))
    q.put(invert_mapper[gpu]) 

# # Change loop
Parallel(n_jobs=N_GPU, backend="threading")( delayed(runner)(experiments[i]) for i in tqdm(range(len(experiments))))

path = model_checkpoint + "/result_splits/"
if not os.path.exists(path):
    os.makedirs(path)

# Combine results
all_results = {}
for t in split_list:
    start_ind, end_ind = t
    with open(model_checkpoint + "/result_splits/" + str(start_ind) + '_' + str(end_ind) + '.json', 'r') as json_file:
        obj_split = json.load(json_file)
        all_results.update(obj_split)

with open(model_checkpoint + "/all_results.json", 'w') as outfile:
    json.dump(all_results, outfile)

print("total predictions: {} dialogues".format(len(all_results)))
