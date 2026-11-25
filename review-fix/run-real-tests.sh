#!/bin/bash
export PATH=/home/vipuser/anaconda3/envs/paddleformers/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/baidu_98_minicpm4_1/PaddleFormers
export CUDA_VISIBLE_DEVICES=0 RUN_SLOW_TEST=True
export MINICPM_LEGACY_CHECKPOINT=/data/minicpm4_0.5b MINICPM4_1_CHECKPOINT=/data/minicpm4.1-8b
cd /root/baidu_98_minicpm4_1/PaddleFormers
python -c "import paddle,pytest;paddle.set_device('gpu:0');raise SystemExit(pytest.main(['-q','tests/transformers/auto/test_minicpm.py','-k','RealCheckpoint','--junitxml=/root/pr4840-review-fix/real-checkpoints.xml']))"
echo $? > /root/pr4840-review-fix/real-checkpoints.exit
