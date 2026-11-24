#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 FLAGS_allow_tf32_cublas=0 FLAGS_allow_tf32_cudnn=0 TOKENIZERS_PARALLELISM=false
export PATH=/home/vipuser/anaconda3/envs/paddleformers/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/root/pr4495-validation:/root/baidu_98_internlm/PaddleFormers
export ALIGN_BACKEND=paddle ALIGN_TRACE=/root/pr4495-validation/paddle-trace
cd /root/baidu_98_internlm/PaddleFormers
python -u -m paddleformers.cli.launcher train /root/pr4495-validation/paddle-training.yaml > /root/pr4495-validation/paddle-training.log 2>&1
echo $? > /root/pr4495-validation/paddle-training.exit
