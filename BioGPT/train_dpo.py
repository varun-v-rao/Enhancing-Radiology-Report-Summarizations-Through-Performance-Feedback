import torch
from transformers import AutoTokenizer, BioGptForCausalLM, AutoModelForCausalLM
from datasets import load_dataset
from transformers import TrainingArguments
from trl import DPOTrainer

model_checkpoint = "/nfs/turbo/umms-vgvinodv/models/finetuned-checkpoints/radsum/biogpt-mimic-cxr/checkpoint-7620"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
model_ref = AutoModelForCausalLM.from_pretrained(model_checkpoint)

dataset = load_dataset("varuUM/mimic-cxr-dpo-with-metrics", split="train")

def filter_func(examples):
    return examples['F1RadGraph'] < 0.4

dpo_dataset = dataset.filter(filter_func)
dpo_dataset = dpo_dataset.remove_columns(["rougeL","F1RadGraph","F1CheXbert"])

dataset_config = "mimic-cxr"
model_name = "biogpt"
save_path: str="/nfs/turbo/umms-vgvinodv/models/finetuned-checkpoints/radsum"
save_path = f"{save_path}/{model_name}-dpo-{dataset_config}"

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    num_train_epochs = 5,
    max_steps =-1,
    save_strategy = 'epoch',
    save_total_limit = 1,
    logging_strategy ='steps',
    logging_steps=20,
    learning_rate=1e-4,
    output_dir=save_path,
    remove_unused_columns=False,
    run_name="dpo_biogpt",
    overwrite_output_dir = True,
)

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.3,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    max_prompt_length=512,
    max_length=1024,
)

dpo_trainer.train()