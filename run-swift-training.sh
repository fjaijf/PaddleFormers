#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 TOKENIZERS_PARALLELISM=false
export PATH=/data/envs/ms-swift/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/root/pr4495-validation/reference-deps:/root/pr4495-validation
export ALIGN_TRACE=/root/pr4495-validation/swift-trace
cd /root/pr4495-validation
swift sft --model /data/models/InternLM-7B --model_type internlm --train_type lora --lora_rank 8 --lora_alpha 16 --lora_dropout 0 --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --torch_dtype float32 --attn_impl eager --dataset /root/pr4495-validation/swift-dataset.jsonl --template internlm_paddle_alignment --external_plugins /root/pr4495-validation/swift-plugin.py --split_dataset_ratio 0 --dataset_shuffle false --train_dataloader_shuffle false --dataset_num_proc 1 --dataloader_num_workers 0 --max_length 512 --max_steps 20 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 1e-4 --lr_scheduler_type constant --warmup_steps 0 --weight_decay 0 --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 --max_grad_norm 0 --gradient_checkpointing false --bf16 false --fp16 false --tf32 false --seed 23 --data_seed 23 --logging_steps 1 --save_strategy no --eval_strategy no --report_to none --optim adamw_torch --disable_tqdm true --output_dir /root/pr4495-validation/swift-training-output > /root/pr4495-validation/swift-training.log 2>&1
echo $? > /root/pr4495-validation/swift-training.exit
