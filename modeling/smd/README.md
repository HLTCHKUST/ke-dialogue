# Knowledge Embedded (KE) on SMD dataset
## Finetune GPT2 model on SMD training set
``` console
SMD_gpt2_graph_False_adj_False_edge_False_unilm_False_flattenKB_False_historyL_1000000000_lr_6.25e-05_epoch_10_weighttie_False_kbpercentage_0_layer_12
❱❱❱ python main.py --
```
## Further adaptation to test set
``` console
❱❱❱ python evaluate_finetune.py --dataset SMD --model_checkpoint runs/SMD_gpt2_graph_False_adj_False_edge_False_unilm_False_flattenKB_False_historyL_1000000000_lr_6.25e-05_epoch_10_weighttie_False_kbpercentage_0_layer_12 --top_k 1 --eval_indices {},{} --filter_domain {}
```
You can also speed up the finetuning process by running experiments parallelly. Please modify the GPU 
``` console
❱❱❱ python evaluate_finetune.py --dataset SMD --model_checkpoint {} --top_k 1 --eval_indices {},{} --filter_domain {}
```