# InternLM-7B precision alignment for PaddleFormers PR #4495

Validated model implementation: `92b8450ea10cd7a7f3a73a21cdd8cbcaa9481bc0`.
Date: 2026-11-24 (Asia/Shanghai). Hardware: one NVIDIA A100-SXM4-80GB.

This branch contains measurement artifacts only. The model implementation remains on the PR branch.

## Results

- FP32 forward: compare `[1, 20, 103168]` logits on identical 31-token input. Max absolute difference `0.00336456298828125`, mean absolute difference `0.00033617028342583044`, RMSE `0.0008801202325685175`. Top-1 token agrees at all 20 positions.
- Greedy generation with KV cache: all 10 newly generated token IDs agree.
- Official training: `python -m paddleformers.cli.launcher train paddle-training.yaml` versus `swift sft`, FP32 LoRA + AdamW, 20 optimizer steps. Max absolute step loss difference `2.47955322265625e-05`, mean absolute difference `8.988380432128907e-06`.
- Step 1: PaddleFormers `2.8029143810272217`, ms-swift `2.802910089492798`.
- Step 20: PaddleFormers `2.745792865753174`, ms-swift `2.7457902431488037`.
- Arithmetic mean of the 20 step losses: PaddleFormers `3.231217074394226`, ms-swift `3.2312176704406737`.

![Loss comparison](loss-comparison.png)
![Signed loss difference](loss-diff.png)

## Training protocol

Both trainers load the same original `internlm/internlm-7b` HF checkpoint (FP16 storage, promoted to FP32 for execution). AMP, TF32, dropout and gradient checkpointing are disabled. Attention uses the eager implementation. Frozen base model and trainable LoRA parameters are FP32.

The data is the 10 examples in `tests/fixtures/dummy/sft/train.jsonl` at the validated PaddleFormers commit, repeated for 2 epochs in their original order. `paddle-dataset.jsonl` and `swift-dataset.jsonl` contain the same conversations in each trainer's supported format. No shuffling, packing, evaluation or truncation occurs. Actual sequence lengths are 24–83 tokens; configured maximum is 512. Batch size and gradient accumulation are both 1. There are 692 supervised next-token positions across the 20 steps.

The ms-swift plugin registers a template matching PaddleFormers' built-in `default` template exactly, including EOS and newline tokens. PaddleFormers shifts labels in the dataset; the reference HF model shifts them in its forward loss. `batch-audit.json` verifies input IDs, effective labels and positions at every step. The first batches and all loss records are included.

LoRA targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` in all 32 layers: rank 8, alpha 16, scaling 2, dropout 0, 224 modules, 19,988,480 trainable parameters. `alignment_common.py` initializes A with identical per-module deterministic uniform arrays, B with zeros, and transposes A for PyTorch's linear weight convention. The matching `initial-lora.json` manifests verify all initial LoRA arrays. This prevents independent random initializations from confounding the loss comparison.

Both official optimizers use AdamW: learning rate 1e-4, constant schedule, no warmup, betas (0.9, 0.999), epsilon 1e-8, weight decay 0, no gradient clipping. Loss is mean cross-entropy over supervised next-token labels, ignoring -100. Loss values are captured before each optimizer update.

`sitecustomize.py` and `swift-plugin.py` only synchronize adapter initialization and observe actual model inputs/losses; neither replaces either framework's loss, backward pass, optimizer or training loop. The final logits diagnostic only records an additional tensor.

**PaddleFormers configuration detail:** explicitly set `compute_type: float32` and `amp_master_grad: false`; setting `bf16: false` alone is insufficient because `compute_type` defaults to BF16. Only the final verified FP32 trial is reported here. Earlier BF16 diagnostic trials are excluded.

## Reproduction

These scripts preserve the absolute paths used on the validation machine. Change `/root/pr4495-validation`, the PaddleFormers checkout path, model path and environment executables to your local locations before running them.

1. Check out the validated PaddleFormers commit and obtain the original InternLM-7B checkpoint, including the original HF model/configuration/tokenizer Python sources and tokenizer.model.
2. Use separate Paddle and PyTorch/ms-swift environments. Versions are recorded in `environment.json`. The old InternLM tokenizer requires SentencePiece 0.1.99 in the reference environment; the installed 0.2.2 rejected the vocabulary's null-character token. On this machine 0.1.99 was installed to the isolated `reference-deps` directory, leaving the shared environment unchanged.
3. Place these files under the validation directory. Run `bash run-inference.sh`, then `bash run-paddle-training.sh`, then `bash run-swift-training.sh` sequentially on the same GPU. Start with empty trace directories to avoid appending loss records from a previous trial.
4. Run `python analyze.py` after copying all 20 batch trace NPZs from both runs. This audits actual batches and regenerates the CSV, summaries and both figures. The artifacts branch includes the first batch for inspection; rerunning the capture scripts produces every batch.

## Scope

This is a short LoRA training alignment run, not full-parameter FP32 AdamW training or a convergence benchmark. Different samples cause the step loss to fluctuate. All tested sequences are below 2048 positions. Long-context dynamic NTK behavior and multi-GPU pipeline training are not established by this experiment; separate unit regression coverage exists for dynamic RoPE registration/extension.

`loss-comparison.csv`, both `losses.jsonl` files and `training-summary.json` are the full-precision numerical records. `loss-diff.png` plots PaddleFormers minus ms-swift, without smoothing.
