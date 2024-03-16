from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler
from accelerate import Accelerator
import numpy as np
import evaluate
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm.auto import tqdm
import datetime
import sys
import cv2
### hyperparameters
train_batch_size=16
learning_rate=5e-5
num_epochs = 1

# task = "matching"
task = sys.argv[1]
# task = "explanation"
eval_model_path = f"/mmfs1/home/jrl712/amazon_home/nlp_final_project/{task}2/{sys.argv[2]}"


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# load train/val/test splits for each task
dset = load_dataset("jmhessel/newyorker_caption_contest", task)
# dset = load_dataset("jmhessel/newyorker_caption_contest", "ranking")
# dset = load_dataset("jmhessel/newyorker_caption_contest", "explanation")

val_dataset = dset["test"].with_format("torch")

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")
def tokenize(seqs):
    return tokenizer(seqs, padding="max_length", truncation=True, max_length=512)

# Load model directly
model = AutoModelForSeq2SeqLM.from_pretrained(eval_model_path)
# model.state_dict = torch.load(eval_model_path)
model.to(device)

def collate_tokenize(data):
  text_batch = [element["from_description"] for element in data]
  label_batch = [element["label"] for element in data]
  tokenized_text = tokenizer(text_batch, padding='longest', truncation=True, return_tensors='pt')
  tokenized_label = tokenizer(label_batch, padding='longest', truncation=True, return_tensors='pt')
#   print(max([len(s) for s in text_batch]))
#   print(tokenized_text["input_ids"].shape)
  return {"text": tokenized_text.to(device), "labels": tokenized_label["input_ids"].squeeze(-1).to(device), "image":[element["image"] for element in data], "caption":[element["caption_choices"] for element in data]}
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=8, collate_fn=collate_tokenize)

metric = evaluate.load("accuracy")
model.eval()

image_id = 0
for batch in val_dataloader:

    if task == "explanation":
        if task == "explanation":
            predictions = model.generate(
            labels=batch["labels"],
            **(batch["text"]),
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            max_new_tokens=100
        )
        for i in range(len(batch["image"])):
            print(batch["image"][i].shape)
            cv2.imwrite(f"/mmfs1/home/jrl712/amazon_home/nlp_final_project/explanation_outs/{image_id}.png",batch["image"][i].cpu().numpy())
            with open(f"/mmfs1/home/jrl712/amazon_home/nlp_final_project/explanation_outs/{image_id}.txt", "w") as f:
                f.write(batch["caption"][i] + "\n" + " ".join(tokenizer.batch_decode(predictions[i])))
            image_id += 1
    else:
        with torch.no_grad():
            print(batch["labels"].shape)
            outputs = model(labels=batch["labels"], **(batch["text"]))
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
#     print(tokenizer.batch_decode(predictions))
#     print(predictions.shape, batch["labels"].shape)
#     print(predictions.dtype, batch["labels"].dtype)
#     print(predictions.device, batch["labels"].device)
#     print(tokenizer.batch_decode(predictions))
#     print(metric.inputs_description)
#     metric.add_batch(predictions=predictions.to(torch.int32).cpu()[:, 0], references=batch["labels"].to(torch.int32).cpu()[:, 0])
# print(metric.compute())