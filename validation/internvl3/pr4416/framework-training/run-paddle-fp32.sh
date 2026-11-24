#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 FLAGS_allow_tf32_cublas=0 FLAGS_allow_tf32_cudnn=0
export PATH=/home/vipuser/anaconda3/envs/paddleformers/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PRECISION_TRACE_BACKEND=paddle
export PRECISION_TRACE_DIR=/root/pr4416-validation/framework-training/paddle-fp32-trace
export PYTHONPATH=/root/pr4416-validation/framework-training/paddle-observer:/root/baidu_98_internvl/PaddleFormers:${PYTHONPATH:-}
cd /root/baidu_98_internvl/PaddleFormers/examples
python -u -m paddleformers.cli.launcher train /root/pr4416-validation/framework-training/paddle-fp32.yaml > /root/pr4416-validation/framework-training/paddle-fp32.log 2>&1
echo $? > /root/pr4416-validation/framework-training/paddle-fp32.exit
