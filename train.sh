#!/bin/bash

# 设置环境变量
export OPENAI_API_KEY=""
export HUGGINGFACE_HUB_CACHE=""

# 设置详细日志
export LOGLEVEL=INFO  # 或者使用DEBUG获取更详细的日志


# 配置accelerate
CFG_FILE=local_ds8.yaml


# 运行训练脚本
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file "$CFG_FILE" \
  RL/train_trl.py \
  --data_path data/simple_data.jsonl \
  --openai_api_key \
  --google_api_key \
  --output_dir checkpoints/r1_grpo_turbo\
  --max_steps 2000 \
  --batch_size 1 \
  --num_generations 4 \
  --max_new_tokens 2000 \
  --gradient_accumulation_steps 4 \
  --lr 1e-5 \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --use_wandb \
  --wandb_api_key  \
  --score_mode continuous \
  --curriculum_steps 0 \
  