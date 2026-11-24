#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 MAX_NUM=1
export PATH=/data/envs/ms-swift/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PRECISION_TRACE_DIR=/root/pr4416-validation/framework-training/swift-trace
export TOKENIZERS_PARALLELISM=false
cd /root/pr4416-validation/framework-training
swift sft --model /root/pr4416-validation/framework-training/model-deterministic --model_type internvl3 --train_type full --torch_dtype float32 --attn_impl eager --dataset /root/pr4416-validation/framework-training/swift-dataset.jsonl --template internvl3_paddle_alignment --external_plugins /root/pr4416-validation/framework-training/swift-plugin.py --split_dataset_ratio 0 --dataset_shuffle false --train_dataloader_shuffle false --dataset_num_proc 1 --dataloader_num_workers 0 --max_length 4096 --max_steps 20 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 1e-6 --lr_scheduler_type constant --warmup_steps 0 --weight_decay 0 --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 --max_grad_norm 0 --gradient_checkpointing false --freeze_vit true --freeze_aligner true --freeze_llm false --bf16 false --fp16 false --tf32 false --seed 23 --data_seed 23 --logging_steps 1 --save_strategy no --eval_strategy no --report_to none --optim adamw_torch --disable_tqdm true --output_dir /root/pr4416-validation/framework-training/swift-output > /root/pr4416-validation/framework-training/swift-full.log 2>&1
echo $? > /root/pr4416-validation/framework-training/swift-full.exit
