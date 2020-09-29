# Knowledge Embedded (KE) on SMD dataset

## Finetune GPT2 model on SMD training set

``` console
❱❱❱ python main.py --dataset SMD --graph False --adj_path False --edge_list False --unilm False --flatten_KB False --max_history 1000000000 --lr 6.25e-05 --n_epochs 10 --weight_tie False --kbpercentage 0 --layers 12
```

## Adapt fine-tuned GPT-2 model on the test set

``` console
❱❱❱ python evaluate_finetune.py --dataset SMD --model_checkpoint runs/SMD_gpt2_graph_False_adj_False_edge_False_unilm_False_flattenKB_False_historyL_1000000000_lr_6.25e-05_epoch_10_weighttie_False_kbpercentage_0_layer_12 --top_k 1 --eval_indices 0,303 --filter_domain ""
```

You can also speed up the finetuning process by running experiments parallelly. Please modify the GPU setting in #L14 of the [code](https://github.com/HLTCHKUST/ke-dialogue/blob/master/modeling/smd/runner_expe_SMD.py#L14).

``` console
❱❱❱ python runner_expe_SMD.py 
```