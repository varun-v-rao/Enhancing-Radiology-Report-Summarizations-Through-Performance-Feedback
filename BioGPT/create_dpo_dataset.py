import torch
import datasets
from pathlib import Path
from transformers import AutoTokenizer, BioGptForCausalLM

from huggingface_hub import login

def build_test_dataset(dataset_config, tokenizer, split="test"):
    data_path = '/nfs/turbo/umms-vgvinodv/data/bioNLP23-Task-1B/data/'
    findings_file_path = Path(data_path).joinpath(dataset_config).joinpath(split+'.findings.tok')
    impression_file_path = Path(data_path).joinpath(dataset_config).joinpath(split+'.impression.tok')


    findings = [line.strip() for line in open(findings_file_path).readlines()]
    impression = [line.strip() for line in open(impression_file_path).readlines()]

    dataset = datasets.Dataset.from_dict({"text":findings,"summary":impression}) 
    
    return dataset

def generate_summary(sample):
    texts = sample["text"]
    summaries = sample["summary"]
    prompt = "The main impression based on the given FINDINGS section of the chest X-ray report are:"

    def generate_input(_text):
        return " ".join([_text,prompt])

    inputs = generate_input(texts) 
    model_inputs = tokenizer(inputs, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        response = tokenizer.decode(model.generate(**model_inputs, max_new_tokens=512)[0], skip_special_tokens=True)
    formatted_response = response.split(":")[-1].strip()
        
    return {
        "prompt": inputs,
        "chosen":summaries,
        "rejected": formatted_response
    }

model_checkpoint = "/nfs/turbo/umms-vgvinodv/models/finetuned-checkpoints/radsum/biogpt-mimic-cxr/checkpoint-7620"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = BioGptForCausalLM.from_pretrained(model_checkpoint)

model.eval()
model.to("cuda")

train_dataset = build_test_dataset('mimic-cxr',tokenizer,'train')
print(f'Number of samples: {len(train_dataset)}')

dpo_dataset = train_dataset.map(generate_summary, remove_columns=list(train_dataset.features))

login(token="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")

dpo_dataset.push_to_hub("varuUM/mimic_cxr_dpo")

'''
import evaluate
from radgraph import F1RadGraph
from f1chexbert import F1CheXbert

rouge = evaluate.load('rouge')
f1radgraph = F1RadGraph(reward_level="partial")
f1chexbert = F1CheXbert(device="cuda")

def compute_metric(examples):
    pred_str = examples["rejected"]
    label_str = examples["chosen"]
    examples["rougeL"] = rouge.compute(predictions=pred_str, references=label_str, rouge_types=['rougeL'],  use_aggregator=False)['rougeL']
    examples["F1RadGraph"] = f1radgraph(hyps=pred_str,refs=label_str)[1]
    examples["F1CheXbert"] = list(f1chexbert(hyps=pred_str,refs=label_str)[1])
    return examples

processed_dataset = dpo_dataset.map(compute_metric, batched=True)

login(token="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")

processed_dataset.push_to_hub("varuUM/mimic-cxr-dpo-with-metrics")

'''