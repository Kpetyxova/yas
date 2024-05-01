import torch
from unsloth import FastLanguageModel
import argparse
from datasets import load_dataset
from sacrebleu.metrics import BLEU
import wandb
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from transformers.utils import logging
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--configuration", type=str, default="ZS")
parser.add_argument("-g", "--grounding", type=str, default="KG")
parser.add_argument("-m", "--mode", type=str, default="predict")

configuration = parser.parse_args().configuration
grounding = parser.parse_args().grounding
mode = parser.parse_args().mode

max_seq_length = 256
dtype = None
load_in_4bit = True

mapping = {}
if configuration == "ZS":
    mapping["model_name"] = "unsloth/tinyllama-bnb-4bit"
    mapping["test_data"] = f"data/test_focus_ZS_{grounding}.jsonl"

elif configuration == "FT":
    mapping["test_data"] = f"data/test_focus_{grounding}.jsonl"
    if grounding == "KG":
        mapping["model_name"] = "data/KG/outputs/checkpoint-1000"
        mapping["train_data"] = "data/train_focus_KG.jsonl"
        mapping["eval_data"] = "data/valid_focus_KG.jsonl"
    elif grounding == "PG":
        mapping["model_name"] = "data/PG/outputs/checkpoint-500"
        mapping["train_data"] = "data/train_focus_PG.jsonl"
        mapping["eval_data"] = "data/valid_focus_PG.jsonl"
    elif grounding == "KG_PG":
        mapping["model_name"] = "data/KG_PG/outputs/checkpoint-500"
        mapping["train_data"] = "data/train_focus_KG_PG.jsonl"
        mapping["eval_data"] = "data/valid_focus_KG_PG.jsonl"
    elif grounding == "vanilla":
        mapping["model_name"] = "data/vanilla/outputs/checkpoint-500"
        mapping["train_data"] = "data/train_focus_vanilla.jsonl"
        mapping["eval_data"] = "data/valid_focus_vanilla.jsonl"

if mode == "predict":
    mapping["model_name"] = "unsloth/tinyllama-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mapping["model_name"],
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
EOS_TOKEN = tokenizer.eos_token

alpaca_prompt = """Below is an instruction that describes a task, \
paired with an input that provides further context. Write a response \
that appropriately completes the request.\n\n### Instruction:\n{}\n\n\
### Input:\n{}\n\n### Response:\n{}"""
bleu = BLEU()


def formatting_prompts_func(examples):
    """Format the prompts for the model."""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


def compute_bleu(eval_preds):
    """Compute BLEU score for the model."""
    logits, labels = eval_preds

    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    predictions = torch.argmax(logits, dim=-1)

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().tolist()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().tolist()

    decoded_preds = [
        tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True) for pred in predictions
    ]
    decoded_labels = [
        tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=True) for label in labels
    ]
    bleu_score = bleu.corpus_score(decoded_preds, [[label] for label in decoded_labels])
    return {"bleu": bleu_score.score}


def read_jsonl(filename):
    """Read a JSONL file."""
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl_file(file, data):
    """Write a JSONL file."""
    with open(file, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def main():
    """Main function for training and inference."""
    if mode == "train":
        if configuration == "FT":
            dataset = load_dataset("json", data_files={"train": mapping["train_data"]}, split="train")
            eval_dataset = load_dataset("json", data_files={"validation": mapping["valid_data"]}, split="validation")
            dataset = dataset.map(formatting_prompts_func, batched=True)
            eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
            wandb.login()
            logging.set_verbosity_info()
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                eval_dataset=eval_dataset,
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                dataset_num_proc=2,
                packing=True,
                compute_metrics=compute_bleu,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=10,
                        early_stopping_threshold=0.0,
                        #  early_stopping_metric="bleu"
                    )
                ],
                args=TrainingArguments(
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=1,
                    gradient_accumulation_steps=4,
                    warmup_ratio=0.1,
                    num_train_epochs=1,
                    learning_rate=2e-5,
                    fp16=not torch.cuda.is_bf16_supported(),
                    bf16=torch.cuda.is_bf16_supported(),
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=0.1,
                    lr_scheduler_type="linear",
                    seed=3407,
                    output_dir="/".join(mapping["model_name"].split("/")[:-1]),
                    evaluation_strategy="steps",
                    # eval_accumulation_steps=4,
                    eval_steps=50,
                    report_to="wandb",
                    load_best_model_at_end=True,
                    save_total_limit=3,
                    run_name=grounding,
                    metric_for_best_model="bleu",
                ),
            )
            wandb.init(project="yas", name=grounding)
            trainer.train()
            wandb.finish()
        else:
            raise ValueError("Invalid configuration for training.")

    else:
        tokenizer.padding_side = "left"
        FastLanguageModel.for_inference(model)
        test = read_jsonl(mapping["test_data"])
        batch_size = 100
        for i in tqdm(range(0, len(test), batch_size)):
            inputs = tokenizer(
                [
                    alpaca_prompt.format(
                        ex["instruction"],
                        ex["input"],
                        "",
                    )
                    for ex in test[i:i+batch_size]
                ],
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).to("cuda")

            outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
            res = tokenizer.batch_decode(outputs)
            for j, t in enumerate(test[i:i+batch_size]):
                if configuration == "FT" and "PG" in grounding:
                    persona_scores, output_text = (
                        res[j].split("### Response:\n")[-1].split("</s>")[0].split("\n", maxsplit=1)
                    )
                    persona_scores = persona_scores.strip("SCORES: ").strip(" ").split(",")
                    persona_scores = [int(p) for p in persona_scores if p]
                    output_text = output_text.replace("\n", " ")
                    t["output_generated"] = output_text
                    t["predicted_personas"] = persona_scores
                else:
                    output_text = res[j].split("### Response:\n")[-1].split("</s>")[0]
                    output_text = output_text.replace("\n", " ")
                    t["output_generated"] = output_text

        write_jsonl_file("data/test_focus_{configuration}_{grounding}_generated.jsonl", test)


if __name__ == "__main__":
    main()
