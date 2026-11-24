#!/bin/bash
mkdir -p /root/pr4416-validation/fp32
export CUDA_VISIBLE_DEVICES=0
export PATH=/home/vipuser/anaconda3/envs/paddleformers/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export FLAGS_allow_tf32_cublas=0
export FLAGS_allow_tf32_cudnn=0
export NVIDIA_TF32_OVERRIDE=0
cd /root/pr4416-validation
python -u measure-fp32.py paddle > fp32/paddle.log 2>&1
echo $? > fp32/paddle.exit
/root/pr4416-validation/reference-env/bin/python -u measure-fp32.py torch > fp32/reference.log 2>&1
echo $? > fp32/reference.exit
