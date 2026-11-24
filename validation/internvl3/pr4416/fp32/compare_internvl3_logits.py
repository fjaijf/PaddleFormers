#!/usr/bin/env python3
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

"""Compare InternVL3 logits from Torch and Paddle on identical inputs.

Example:
  # 1) Run in an environment with torch/transformers installed.
  python examples/inference/compare_internvl3_logits.py \
      --backend torch \
      --model_name_or_path /home/Userlist/guhexiang/internvl3-1b \
      --torch_npz /tmp/internvl3_torch_logits.npz

  # 2) Run in a Paddle environment. The torch npz also carries input arrays.
  python examples/inference/compare_internvl3_logits.py \
      --backend paddle \
      --model_name_or_path /home/Userlist/guhexiang/internvl3-1b \
      --torch_npz /tmp/internvl3_torch_logits.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
from PIL import Image


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["torch", "paddle", "both", "compare"],
        required=True,
        help="torch saves reference logits, paddle saves and compares, both runs both in one process.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="/home/Userlist/guhexiang/internvl3-1b",
        help="Local InternVL3 HF/PaddleFormers model directory.",
    )
    parser.add_argument(
        "--image_path",
        default="/home/Userlist/guhexiang/internvl3-1b/examples/image1.jpg",
        help="Input image used when the torch npz does not already contain prepared inputs.",
    )
    parser.add_argument("--question", default="Please describe the image shortly.")
    parser.add_argument(
        "--torch_npz",
        default="/tmp/internvl3_torch_logits.npz",
        help="Reference npz path written by the torch backend.",
    )
    parser.add_argument(
        "--paddle_npz",
        default="/tmp/internvl3_paddle_logits.npz",
        help="Paddle npz path written by the paddle backend.",
    )
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, gpu, or gpu:0.")
    parser.add_argument("--num_tokens", type=int, default=10, help="Number of token positions to compare.")
    parser.add_argument("--logit_start", type=int, default=0, help="First token position used for comparison.")
    parser.add_argument("--threshold", type=float, default=1e-2, help="Mean absolute diff threshold.")
    parser.add_argument("--max_dynamic_patch", type=int, default=None)
    parser.add_argument("--min_dynamic_patch", type=int, default=None)
    parser.add_argument("--use_thumbnail", type=str2bool, default=None)
    parser.add_argument(
        "--tokenizer_backend",
        choices=["auto", "hf", "paddle"],
        default="auto",
        help="Tokenizer used only when prepared inputs are not loaded from torch_npz.",
    )
    parser.add_argument("--trust_remote_code", type=str2bool, default=True)
    parser.add_argument("--use_flash_attn", type=str2bool, default=False)
    parser.add_argument("--convert_from_hf", type=str2bool, default=True)
    parser.add_argument(
        "--load_checkpoint_format",
        default="flex_checkpoint",
        choices=["flex_checkpoint", "unified_checkpoint", "sharding_io", "none"],
    )
    parser.add_argument("--load_via_cpu", type=str2bool, default=True)
    return parser.parse_args()


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def ensure_repo_on_path():
    root = repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def load_config(model_path: str) -> dict[str, Any]:
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def get_num_image_token(config: dict[str, Any]) -> int:
    image_size = config.get("force_image_size") or config["vision_config"]["image_size"]
    patch_size = config["vision_config"]["patch_size"]
    downsample_ratio = config["downsample_ratio"]
    return int((image_size // patch_size) ** 2 * (downsample_ratio**2))


def get_image_size(config: dict[str, Any]) -> int:
    return int(config.get("force_image_size") or config["vision_config"]["image_size"])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def image_to_chw(image: Image.Image, image_size: int) -> np.ndarray:
    image = image.convert("RGB").resize((image_size, image_size), resample=BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return array.transpose(2, 0, 1)


def load_pixel_values(args, config: dict[str, Any]) -> np.ndarray:
    image_size = get_image_size(config)
    min_num = args.min_dynamic_patch or int(config.get("min_dynamic_patch", 1))
    max_num = args.max_dynamic_patch or int(config.get("max_dynamic_patch", 12))
    use_thumbnail = bool(config.get("use_thumbnail", False)) if args.use_thumbnail is None else args.use_thumbnail
    image = Image.open(args.image_path).convert("RGB")
    images = dynamic_preprocess(
        image,
        min_num=min_num,
        max_num=max_num,
        image_size=image_size,
        use_thumbnail=use_thumbnail,
    )
    return np.stack([image_to_chw(img, image_size) for img in images]).astype("float32")


def build_prompt(question: str, num_image_token: int, num_patches: int) -> str:
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
    if "<image>" in question:
        return question.replace("<image>", image_tokens, 1)
    return f"{image_tokens}\n{question}"


def tokenize_with_hf(model_path: str, prompt: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )
    encoded = tokenizer(prompt, return_tensors="np")
    image_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return encoded["input_ids"], encoded["attention_mask"], int(image_token_id)


def tokenize_with_paddle(model_path: str, prompt: str):
    ensure_repo_on_path()
    from paddleformers.transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded = tokenizer(prompt, return_tensors="np")
    image_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return encoded["input_ids"], encoded["attention_mask"], int(image_token_id)


def tokenize_prompt(args, prompt: str, prefer_hf: bool):
    if args.tokenizer_backend == "hf":
        return tokenize_with_hf(args.model_name_or_path, prompt, args.trust_remote_code)
    if args.tokenizer_backend == "paddle":
        return tokenize_with_paddle(args.model_name_or_path, prompt)

    backends = ("hf", "paddle") if prefer_hf else ("paddle", "hf")
    errors = []
    for backend in backends:
        try:
            if backend == "hf":
                return tokenize_with_hf(args.model_name_or_path, prompt, args.trust_remote_code)
            return tokenize_with_paddle(args.model_name_or_path, prompt)
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            errors.append(f"{backend}: {exc}")
    raise RuntimeError("Could not tokenize prompt. " + " | ".join(errors))


def prepare_inputs(args, prefer_hf: bool) -> dict[str, Any]:
    config = load_config(args.model_name_or_path)
    num_image_token = get_num_image_token(config)
    pixel_values = load_pixel_values(args, config)
    prompt = build_prompt(args.question, num_image_token, pixel_values.shape[0])
    input_ids, attention_mask, image_token_id = tokenize_prompt(args, prompt, prefer_hf=prefer_hf)
    image_flags = np.ones((pixel_values.shape[0], 1), dtype="int64")
    position_ids = np.cumsum(attention_mask.astype("int64"), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0).astype("int64")

    seq_len = int(input_ids.shape[1])
    if args.logit_start < 0 or args.num_tokens <= 0 or args.logit_start + args.num_tokens > seq_len:
        raise ValueError(
            f"Invalid logit slice [{args.logit_start}, {args.logit_start + args.num_tokens}) for seq_len={seq_len}."
        )
    return {
        "input_ids": input_ids.astype("int64"),
        "attention_mask": attention_mask.astype("int64"),
        "position_ids": position_ids,
        "pixel_values": pixel_values,
        "image_flags": image_flags,
        "image_token_id": np.asarray(image_token_id, dtype="int64"),
        "prompt": np.asarray(prompt),
        "num_image_token": np.asarray(num_image_token, dtype="int64"),
        "num_patches": np.asarray(pixel_values.shape[0], dtype="int64"),
        "logit_start": np.asarray(args.logit_start, dtype="int64"),
        "num_tokens": np.asarray(args.num_tokens, dtype="int64"),
    }


def load_inputs_from_npz(path: str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    required = ["input_ids", "attention_mask", "pixel_values", "image_flags", "image_token_id"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing prepared input arrays: {missing}")
    return {key: data[key] for key in data.files if key != "logits"}


def mkdir_for_file(path: str):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_backend_npz(path: str, backend: str, logits: np.ndarray, inputs: dict[str, Any]):
    mkdir_for_file(path)
    payload = dict(inputs)
    payload["backend"] = np.asarray(backend)
    payload["logits"] = logits.astype("float32")
    np.savez(path, **payload)
    print(f"[{backend}] saved {path}")
    print(f"[{backend}] logits shape: {tuple(logits.shape)}")


def torch_dtype_from_name(torch, dtype_name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def resolve_torch_device(torch, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("gpu"):
        return device.replace("gpu", "cuda", 1)
    return device


def run_torch(args, inputs: dict[str, Any]) -> np.ndarray:
    import torch
    from transformers import AutoModel

    device = resolve_torch_device(torch, args.device)
    dtype = torch_dtype_from_name(torch, args.dtype)
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=args.use_flash_attn,
        trust_remote_code=args.trust_remote_code,
    ).eval()
    model.to(device)
    model.img_context_token_id = int(inputs["image_token_id"])

    input_ids = torch.from_numpy(inputs["input_ids"]).to(device=device)
    attention_mask = torch.from_numpy(inputs["attention_mask"]).to(device=device)
    position_ids = None
    if "position_ids" in inputs:
        position_ids = torch.from_numpy(inputs["position_ids"]).to(device=device)
    pixel_values = torch.from_numpy(inputs["pixel_values"]).to(device=device, dtype=dtype)
    image_flags = torch.from_numpy(inputs["image_flags"]).to(device=device)

    start = int(inputs.get("logit_start", np.asarray(args.logit_start)))
    end = start + int(inputs.get("num_tokens", np.asarray(args.num_tokens)))
    with torch.inference_mode():
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits[:, start:end, :].float().cpu().numpy()
    return logits


def resolve_paddle_device(paddle, device: str) -> str:
    if device == "auto":
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu:0"
        return "cpu"
    if device.startswith("cuda"):
        return device.replace("cuda", "gpu", 1)
    return device


def run_paddle(args, inputs: dict[str, Any]) -> np.ndarray:
    ensure_repo_on_path()
    import paddle

    paddle.set_device(resolve_paddle_device(paddle, args.device))
    paddle.set_grad_enabled(False)

    from paddleformers.transformers import AutoModelForConditionalGeneration

    load_checkpoint_format = None if args.load_checkpoint_format == "none" else args.load_checkpoint_format
    model = AutoModelForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        dtype=args.dtype,
        convert_from_hf=args.convert_from_hf,
        load_checkpoint_format=load_checkpoint_format,
        load_via_cpu=args.load_via_cpu,
        use_flash_attn=args.use_flash_attn,
    )
    model.eval()
    model.img_context_token_id = int(inputs["image_token_id"])

    input_ids = paddle.to_tensor(inputs["input_ids"], dtype="int64")
    attention_mask = paddle.to_tensor(inputs["attention_mask"], dtype="int64")
    position_ids = paddle.to_tensor(inputs["position_ids"], dtype="int64") if "position_ids" in inputs else None
    pixel_values = paddle.to_tensor(inputs["pixel_values"]).astype(args.dtype)
    image_flags = paddle.to_tensor(inputs["image_flags"], dtype="int64")

    start = int(inputs.get("logit_start", np.asarray(args.logit_start)))
    end = start + int(inputs.get("num_tokens", np.asarray(args.num_tokens)))
    with paddle.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits[:, start:end, :].astype("float32").numpy()
    return logits


def compare_npz(torch_npz: str, paddle_npz: str, threshold: float):
    torch_data = np.load(torch_npz, allow_pickle=False)
    paddle_data = np.load(paddle_npz, allow_pickle=False)
    torch_logits = torch_data["logits"].astype("float32")
    paddle_logits = paddle_data["logits"].astype("float32")
    if torch_logits.shape != paddle_logits.shape:
        raise ValueError(f"Shape mismatch: torch={torch_logits.shape}, paddle={paddle_logits.shape}")

    abs_diff = np.abs(torch_logits - paddle_logits)
    mean_diff = float(abs_diff.mean())
    max_diff = float(abs_diff.max())
    print(f"[diff] mean_abs_diff={mean_diff:.8e}")
    print(f"[diff] max_abs_diff={max_diff:.8e}")
    print(f"[diff] threshold={threshold:.8e}")
    if mean_diff > threshold:
        raise SystemExit(f"FAILED: mean_abs_diff {mean_diff:.8e} > threshold {threshold:.8e}")
    print("PASSED")


def main():
    args = parse_args()

    if args.backend in {"torch", "both"}:
        inputs = prepare_inputs(args, prefer_hf=True)
        torch_logits = run_torch(args, inputs)
        save_backend_npz(args.torch_npz, "torch", torch_logits, inputs)

    if args.backend in {"paddle", "both"}:
        if os.path.exists(args.torch_npz):
            inputs = load_inputs_from_npz(args.torch_npz)
        else:
            inputs = prepare_inputs(args, prefer_hf=False)
        paddle_logits = run_paddle(args, inputs)
        save_backend_npz(args.paddle_npz, "paddle", paddle_logits, inputs)
        if os.path.exists(args.torch_npz):
            compare_npz(args.torch_npz, args.paddle_npz, args.threshold)

    if args.backend == "compare":
        compare_npz(args.torch_npz, args.paddle_npz, args.threshold)


if __name__ == "__main__":
    main()
