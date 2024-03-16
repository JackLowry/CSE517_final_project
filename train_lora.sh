export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/mmfs1/home/jrl712/amazon_home/nlp_final_project/lora_finetune"
# export HUB_MODEL_ID="pokemon-lora"
export DATASET_NAME="jmhessel/newyorker_caption_contest"

accelerate launch /mmfs1/home/jrl712/amazon_home/nlp_final_project/diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataset_config_name="explanation" \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --validation_prompt="He has a summer job as a scarecrow." \
  --seed=1337 \
  --image_column="image" \
  --caption_column="caption_choices"

#--mixed_precision="fp16"  
#     --push_to_hub \
#   --hub_model_id=${HUB_MODEL_ID} \
#   --report_to=wandb \