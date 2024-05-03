from pathlib import Path
import datasets
from datasets import Image
from torchvision import transforms
import os.path


def generate_image_path(line):
    return str(Path(data_path).joinpath(dataset_config).joinpath(line.strip().split(',')[0]))

dataset_config = "mimic-cxr"
split="train"
data_path = '/nfs/turbo/umms-vgvinodv/data/bioNLP23-Task-1B/data/'

findings_file_path = Path(data_path).joinpath(dataset_config).joinpath(split+'.findings.tok')
impression_file_path = Path(data_path).joinpath(dataset_config).joinpath(split+'.impression.tok')
image_file_path = Path(data_path).joinpath(dataset_config).joinpath(split+'.image.tok')


findings = [line.strip() for line in open(findings_file_path).readlines()]
impression = [line.strip() for line in open(impression_file_path).readlines()]
image_paths = [generate_image_path(line) for line in open(image_file_path).readlines()]

dataset = datasets.Dataset.from_dict({"text":findings,"query":image_paths})

def check_img_exists(example):
    return os.path.isfile(example["query"])

dataset = dataset.filter(check_img_exists, num_proc=4)
dataset = dataset.cast_column("query", Image())

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def image_transforms(samples):
    samples["query"] = [transform(image.resize((384,384)).convert("RGB")) for image in samples["query"]]
    return samples
dataset = dataset.map(image_transforms, batched=True, batch_size=256)

dataset.save_to_disk("ppo_dataset.hf")
print(dataset[0])