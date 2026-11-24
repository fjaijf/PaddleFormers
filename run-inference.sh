#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 FLAGS_allow_tf32_cublas=0 FLAGS_allow_tf32_cudnn=0 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/root/pr4495-validation/reference-deps
export PATH=/home/vipuser/anaconda3/envs/paddleformers/bin:/usr/local/cuda/bin:$PATH
/data/envs/ms-swift/bin/python -u /root/pr4495-validation/inference.py torch > /root/pr4495-validation/torch-inference.log 2>&1
echo $? > /root/pr4495-validation/torch-inference.exit
python -u /root/pr4495-validation/inference.py paddle > /root/pr4495-validation/paddle-inference.log 2>&1
echo $? > /root/pr4495-validation/paddle-inference.exit
