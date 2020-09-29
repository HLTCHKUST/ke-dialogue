# Learning Structured Knowledge with Parametersfor Task-Oriented Dialogue Systems

## Overview
We split our scripts into 2 separate folders, which are:
- `knowledge_embed` for dataset generation
- `modeling` for preprocessing, training, and automatic evaluation
- `others` for others scripts such as preparing the dataset, inserting data to `neo4j`, `human evaluation`, and other additional scripts

The provided code is just to get better understanding about the flow of the work and probably it is not runnable. We will publish the complete code on a public source code repository after the work is published.

## Dependencies
We listed our dependencies on `requirements.txt`, you can install the dependencies by running
```
pip install -r requirements.txt
```

In addition, our code also includes `fp16` support with `apex`. You can find the package from https://github.com/NVIDIA/apex.

## Source code overview
#### Knowledge Embed
Inside the `knowledge_embed` folder, we split the scripts per dataset that are used for the experiments. Below are the details of each files inside the folders:

###### BABI5
- `generate_delexicalization_babi.py` | script for generating templates given BABI5 dialogues
- `generate_dialogues_babi.py` | script for generating dialogue from BABI5 templates, knowledge is retrieved from provided knowledge base by using `pandas`
- `revertible_string.py` contains string wrapper class for relexicalization
- `utils.py` contains common functions used in knowledge embedding phase

###### CamRest
- `generate_delexicalization_CAMREST.py` | script for generating templates given CamRest dialogues
- `generate_dialogues_CAMREST.py` | script for generating dialogue from CamRest templates, knowledge is retrieved from provided knowledge base by using `pandas`
- `revertible_string.py` contains string wrapper class for relexicalization
- `utils.py` contains common functions used in knowledge embedding phase

###### SMD
- `generate_delexicalization_SMD.py` | script for generating templates given SMD dialogues. There are a lot of noise in the generated templates, so we need do some manual fix before generating the dialogues
- `generate_dialogues_SMD_sql.py` | script for generating dialogue from SMD templates, knowledge is retrieved from provided knowledge base by using `SQLite`
- `revertible_string.py` contains string wrapper class for relexicalization
- `utils.py` contains common functions used in knowledge embedding phase

###### MWoZ
- `generate_delex_MWOZ_ATTRACTION.py` | script for generating templates given MWoZ dialogues in attraction domain
- `generate_delex_MWOZ_HOTEL.py` | script for generating templates given MWoZ dialogues in hotel domain
- `generate_delex_MWOZ_RESTAURANT.py` | script for generating templates given MWoZ dialogues in restaurant domain
- `generate_delex_MWOZ_TRAIN.py` | script for generating templates given MWoZ dialogues in train domain
- `generate_redelex_augmented_MWOZ.py` | script for generating dialogue from MWoZ templates, knowledge is retrieved from provided knowledge base by using `SQLite`
- `generate_MWOZ_dataset.py` | script for performing normalization and splitting for MWoZ dataset

###### Opendialkg
- `generate_delexicalization_DIALKG.py` | script for generating templates given opendialkg dialogues
- `generate_dialogues_DIALKG.py` | script for generating dialogue from opendialkg templates, knowledge is retrieved from `neo4j` by using `CYPHER` query 
- `revertible_string.py` | string wrapper class for relexicalization
- `dialkg_utils.py` | common functions used in knowledge embedding phase

#### Modeling
Inside the `modeling` folder, we split the scripts per dataset that are used for the experiments. Below are the details of each files inside the folders:

###### BABI5
- `main.py` | script for training the model and output the checkpoint of the last N epochs of the trained model
- `evaluate.py` | script for evaluating the model given the model checkpoint and the test set file. This script will output a generated system response used for scoring
- `scorer_BABI5.py` | script for calculating automatic evaluation score for BABI5 dataset
- `utils` folder containing scripts with common functions used in modelling phase

###### CamRest
- `main.py` | script for training the model and output the checkpoint of the last N epochs of the trained model
- `evaluate.py` | script for evaluating the model given the model checkpoint and the test set file. This script will output a generated system response used for scoring
- `scorer_CAMREST.py` | script for calculating automatic evaluation score for CamRest dataset
- `success_scorer_CAMREST.ipynb` | jupyter notebook containing script for calculating success score for different models on CamRest dataset
- `utils` folder containing scripts with common functions used in modeling phase

###### SMD
- `main.py` | script for training the model and output the checkpoint of the last N epochs of the trained model
- `evaluate.py` | script for evaluating the model given the model checkpoint (GPT and GPT+KB) and the test set file. This script will output a generated system response used for scoring
- `time_evaluate_finetune.py` | script for measuring the finetuning duration on SMD dataset
- `evaluate_finetune.py` | script for fine-tuning and evaluating the fine-tuned model (GPT2+KE) given the model checkpoint and the test set file. This script will output a generated system response used for scoring
- `scorer_SMD.py` | script for calculating automatic evaluation score for SMD dataset
- `utils` folder containing scripts with common functions used in modeling phase

###### MWoZ
- `main.py` | script for training the model and output the checkpoint of the last N epochs of the trained model
- `evaluate.py` | script for evaluating the model given the model checkpoint and the test set file. This script will output a generated system response used for scoring
- `scorer_MWOZ.py` | script for calculating automatic evaluation score for all domains MWoZ dataset
- `scorer_MWOZ_single.py` | script for calculating automatic evaluation score for single domain MWoZ dataset
- `utils` folder containing scripts with common functions used in modeling phase

###### Opendialkg
- `main.py` | script for training the model and output the checkpoint of the last N epochs of the trained model
- `evaluate.py` | script for evaluating the model given the model checkpoint and the test set file. This script will output a generated system response used for scoring
- `scorer_DIALKG.py` | script for calculating automatic evaluation score for Opendialkg dataset
- `utils` folder containing scripts with common functions used in modeling phase

#### Others
Inside the `others` folder, there are severals scripts in different formats. Below are the details of each files inside the folders:
- `setup.sh` | shell script for downloading the dataset
- `load_neo4j.ipynb` | jupyter notebook containing script for injecting OpendialKG graphs into `neo4j`
- `human_eval_script.ipynb` | jupyter notebook containing script for calculating human evaluation score

## Further Details
For the details regarding to the experiments, hyperparameters, and Evaluation results you can find it in the main paper of and suplementary materials of our work.
