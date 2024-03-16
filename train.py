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
### hyperparameters
train_batch_size=16
learning_rate=5e-5
num_epochs = 10

task = sys.argv[1]
# task = "matching"
# task = "ranking"
# task = "explanation"

device = torch.device("cuda")

# load train/val/test splits for each task
dset = load_dataset("jmhessel/newyorker_caption_contest", task)

train_dataset = dset["train"].with_format("torch")

accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")
# Load model directly
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-large")
model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)

def collate_tokenize(data):
  text_batch = [element["from_description"] for element in data]
  label_batch = [element["label"] for element in data]
  tokenized_text = tokenizer(text_batch, padding='longest', truncation=True, return_tensors='pt')
  tokenized_label = tokenizer(label_batch, padding='longest', truncation=True, return_tensors='pt')

  return {"text": tokenized_text, "labels": tokenized_label["input_ids"].squeeze()}
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, collate_fn=collate_tokenize)
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

progress_bar = tqdm(range(num_training_steps))
model.train()
progress = 0
losses = []
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(labels=batch["labels"], **(batch["text"]))
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        if progress % 50 == 0:
            model.save_pretrained(f"{task}2/loss_{loss}_{str(datetime.datetime.now())}", from_pt=True) 
        progress += 1
        losses.append(loss.item())
        progress_bar.update(1)
model.save_pretrained(f"{task}2/loss_{loss}_{str(datetime.datetime.now())}", from_pt=True) 
print(losses)