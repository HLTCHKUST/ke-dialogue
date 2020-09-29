# Knowledge Embedded (KE) on SMD dataset
## Data Setup

## Build databases for SQL query
```console
❱❱❱ python generate_dialogues_SMD.py --build_db --split test
```
## Generate dialogues based on pre-designed templates
We generate dialogues by domains. The following command enables you to generate dialogues in `weather` domain
``` console
❱❱❱ python generate_dialogues_SMD.py --split test --dialogue_path ./templates/weather_template.txt --domain weather --num_augmented_dialogue 100 --output_folder ./data/test
```
