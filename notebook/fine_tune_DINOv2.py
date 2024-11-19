from datasets import load_dataset, enable_caching, disable_caching, is_caching_enabled
# disable_caching()
import os
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize
import torch

# see https://huggingface.co/docs/datasets/image_dataset to load your own custom dataset
# dataset = load_dataset('parquet', data_dir='~/Datasets/oxford-iiit-pet')
dataset = load_dataset("timm/oxford-iiit-pet")


labels = dataset["train"].features["label"].names
id2label = {id: label for id, label in enumerate(labels)}
label2id = {label: id for id, label in enumerate(labels)}
print(id2label)

model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, id2label=id2label, label2id=label2id)

mean = processor.image_mean
std = processor.image_std
interpolation = processor.resample

train_transform = Compose([
    RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=interpolation),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
    ToTensor(),
    Normalize(mean=mean, std=std),
])

def prepare(batch, mode="train"):
  # get images
  images = batch["image"]

  # prepare for the model
  if mode == "train":
    images = [train_transform(image.convert("RGB")) for image in images]
    pixel_values = torch.stack(images)
  elif mode == "test":
    pixel_values = processor(images, return_tensors="pt").pixel_values
  else:
    raise ValueError(f"Mode {mode} not supported")

  inputs = {}
  inputs["pixel_values"] = pixel_values
  inputs["labels"] = torch.tensor(batch["label"])

  return inputs

# set num_proc equal to the number of CPU cores on your machine
# see https://docs.python.org/3/library/multiprocessing.html#multiprocessing.cpu_count
train_dataset = dataset["train"].map(prepare, num_proc=8, batched=True, batch_size=32, fn_kwargs={"mode":"train"})
eval_dataset = dataset["test"].map(prepare, num_proc=8, batched=True, batch_size=32, fn_kwargs={"mode":"test"})
train_dataset.set_format("torch")
eval_dataset.set_format("torch")
train_dataset[0]["pixel_values"].shape
train_dataset[0]["labels"]

from sklearn.metrics import accuracy_score
import numpy as np
from transformers import TrainingArguments, Trainer

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    accuracy = accuracy_score(y_pred=predictions, y_true=eval_pred.label_ids)
    return {"accuracy": accuracy}


args = TrainingArguments(
    f"{model_name}-finetuned-oxford",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)

import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

trainer.train()
