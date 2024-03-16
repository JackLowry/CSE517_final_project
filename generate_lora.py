from diffusers import AutoPipelineForText2Image
import torch

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

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("/mmfs1/home/jrl712/amazon_home/nlp_final_project/lora_finetune/checkpoint-15000", weight_name="pytorch_lora_weights.safetensors")

# load train/val/test splits for each task
dset = load_dataset("jmhessel/newyorker_caption_contest", "explanation")


val_dataset = dset["test"].with_format("torch")

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")


def collate_tokenize(data):
#   text_batch = [element["from_description"] for element in data]
#   label_batch = [element["label"] for element in data]
#   tokenized_text = tokenizer(text_batch, padding='longest', truncation=True, return_tensors='pt')
#   tokenized_label = tokenizer(label_batch, padding='longest', truncation=True, return_tensors='pt')
#   print(max([len(s) for s in text_batch]))
#   print(tokenized_text["input_ids"].shape)
  return {"caption":[element["caption_choices"] for element in data], "image":[element["image"] for element in data]}
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=8, collate_fn=collate_tokenize)


image_id = 0
for batch in val_dataloader:

    images = pipeline(batch["caption"]).images

    for i in range(len(images)):
        cv2.imwrite(f"/mmfs1/home/jrl712/amazon_home/nlp_final_project/lora_outs/{image_id}_orig.png",batch["image"][i].cpu().numpy())
        # cv2.imwrite(f"/mmfs1/home/jrl712/amazon_home/nlp_final_project/lora_outs/{image_id}.png",images[i].cpu().numpy())   
        images[i].save(f"/mmfs1/home/jrl712/amazon_home/nlp_final_project/lora_outs/{image_id}.png")
        with open(f"/mmfs1/home/jrl712/amazon_home/nlp_final_project/lora_outs/{image_id}.txt", "w") as f:
            f.write(batch["caption"][i])
        image_id += 1
#     print(tokenizer.batch_decode(predictions))
#     print(predictions.shape, batch["labels"].shape)
#     print(predictions.dtype, batch["labels"].dtype)
#     print(predictions.device, batch["labels"].device)
#     print(tokenizer.batch_decode(predictions))
#     print(metric.inputs_description)
#     metric.add_batch(predictions=predictions.to(torch.int32).cpu()[:, 0], references=batch["labels"].to(torch.int32).cpu()[:, 0])
# print(metric.compute())