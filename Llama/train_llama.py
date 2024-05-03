import fire
import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers

from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset
from pathlib import Path
import datasets

from transformers import default_data_collator

from transformers import Trainer, TrainingArguments

class Concatenator(object):
    def __init__(self, chunk_size=1024):
        self.chunk_size=chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}
        
    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result
    
def build_dataset(dataset_config, tokenizer, split):
    data_path = '/nfs/turbo/umms-vgvinodv/data/bioNLP23-Task-1B/data/'
    findings_file_path = Path(data_path).joinpath(dataset_config).joinpath(split+'.findings.tok')
    impression_file_path = Path(data_path).joinpath(dataset_config).joinpath(split+'.impression.tok')


    findings = [line.strip() for line in open(findings_file_path).readlines()]
    impression = [line.strip() for line in open(impression_file_path).readlines()]

    dataset = datasets.Dataset.from_dict({"text":findings,"summary":impression}) 
    
    prompt = (
        f"{{text}} The main impression based on the given FINDINGS section of the chest X-ray report are: {{summary}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                text=sample["text"],
                summary=sample["summary"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        num_proc=4,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True, num_proc=4)
    
    return dataset

def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules = ["q_proj", "v_proj"]
    )

    # prepare int-8/int-4 model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

def main(
    model_checkpoint: str="meta-llama/Llama-2-7b-hf",
    seed: int=42,
    batch_size: int=16,
    num_train_epochs: int=1,
    num_proc: int=4,
    save_path: str="/nfs/turbo/umms-vgvinodv/models/finetuned-checkpoints/radsum"
):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    checkpoint = model_checkpoint#"meta-llama/Llama-2-7b-hf"
    dataset_config: str="mimic-cxr"
    model_name = checkpoint.split("/")[-1]
    save_path = f"{save_path}/{model_name}-{dataset_config}"

    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint,       
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training


    train_dataset = build_dataset(dataset_config, tokenizer, 'train')
    eval_dataset = build_dataset(dataset_config, tokenizer, "test")

    print(f'Number of training samples: {len(train_dataset)}')
    print(f'Number of training samples: {len(eval_dataset)}')

    data_collator = default_data_collator

    # create peft config
    model, lora_config = create_peft_config(base_model)


    # Training Args
    training_args = TrainingArguments(
        output_dir=save_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        overwrite_output_dir = True,
        fp16=True,
        #push_to_hub=True,
    )
    #'gradient_accumulation_steps': 4,
    #'gradient_checkpointing': True,


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    fire.Fire(main)