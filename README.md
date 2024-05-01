# You Are Special: Personalized Knowledge Grounding in Language Models

## Data processing
All the files relevant to the project, should be placed to `data` directory. To download them, use the following command:
```
gdown --folder https://drive.google.com/drive/folders/1x-_NbgxpSgcM-WsjMzWgH-NC6oWleixL 
```
If you get "Permission denied", please use the link above to download `data` directory and put it to the root directory of the project.


## Knowledge grounding
To predict relevant knowledge candidates for each dialog turn, run the following, specifying argument `--filename` (the name of a `.json` file from `data` directory).
```
python knowledge_selection.py --filename test_focus.json
```

## Persona grounding and generation
### Paramater-efficient fine-tuning using QLoRA
Before PeFT, reformat the files with functions `data2instructions` and then `configure` from `utils.py`. Please refer to their docstrings for the details.

The script was adapted from the [unsloth notebook](https://colab.research.google.com/drive/1AZghoNBQaMDgWJpi4RbffGM1h6raLUj9?usp=sharing). To fine-tune the model, run the following command, specifying argument `-g`: `KG` for knowledge grounding, `PG` for persona grounding, or `KG_PG` for knowledge and persona grounding:
```
python inference.py -c ZS -g KG_PG -m train
```

### Inference
To run prediction on test, run the following command, specifying arguments `-c` (`FT` for fine-tuning or `ZS` for zero-shot), and `-g` (see fine-tuning section):
```
python inference.py -c FT -g KG_PG -m predict
```

### Evaluation
To evaluate the quality of outputs, run the following command, specifying argument `--filename` (the name of a `.jsonl` file from `data` directory that has fields `output_generated`, `output`, and only for PG `predicted_personas`).
```
python eval.py --filename test_focus_KG_PG_generated.jsonl
```

