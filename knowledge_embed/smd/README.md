# Knowledge Embedded (KE) on SMD dataset
## Data Setup

Download the preprocessed [**dataset**](https://drive.google.com/open?id=1p5FgDcXYPp3s0MzQSbAi-ixqRxNhtfXX) and put it under `./knowledge_embed/smd` folder.

```console
❱❱❱ unzip ./knowledge_embed/smd/SMD.zip
```

## Build databases for SQL query

```console
❱❱❱ python generate_dialogues_SMD.py --build_db --split test
```

## Generate dialogues based on pre-designed templates

We generate dialogues based on pre-designed templates by domains. The following command enables you to generate dialogues in `weather` domain. Please replace `weather` with `navigate` or `schedule` in `dialogue_path` and `domain` arguments if you want to generate dialogues in the other two domains. You can also change number of templates used in relexicalization process by changing the argument `num_augmented_dialogue`.

``` console
❱❱❱ python generate_dialogues_SMD.py --split test --dialogue_path ./templates/weather_template.txt --domain weather --num_augmented_dialogue 100 --output_folder ./SMD/test
```
