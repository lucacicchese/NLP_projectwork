import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent))
from arrow_detection.utils import select_best_cuda_device

torch.backends.cudnn.enabled = False

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
DTYPE = getattr(torch, os.environ.get("TORCH_DTYPE", "float16"))
GRID = 32
NORM_SCALE = 1000
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

PROMPT = (
    "Look for arrowheads in this image. For each one, find its point location as a 2d point, "
    "using the point format convention you already know. Then, from that arrowhead, find its "
    "tail and output its coordinates the same way. Do this for every arrow.\n"
    "Only output the coordinates in the format you know, linking each tail and head pair with a "
    "label: "
    '[{"point_2d": [x, y], "label": "arrow_1_tail"}, {"point_2d": [x, y], "label": "arrow_1_head"}, ...]\n'
    "If there are no arrows, respond with []."
)

PROMPT_NORMALIZED = (
    "Look for arrowheads in this image. For each one, find its point location as a 2d point, "
    "then from that arrowhead find its tail. Do this for every arrow.\n"
    "Express every coordinate on a normalized 0-1000 scale, independent of the image's actual "
    "pixel resolution: x = round(pixel_x / image_width * 1000), y = round(pixel_y / image_height * 1000). "
    "So (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner of the image, "
    "regardless of its real size.\n"
    "Only output the coordinates in this normalized format, linking each tail and head pair with a "
    "label: "
    '[{"point_2d": [x, y], "label": "arrow_1_tail"}, {"point_2d": [x, y], "label": "arrow_1_head"}, ...]\n'
    "If there are no arrows, respond with []."
)

_LABEL_RE = re.compile(r"arrow[\W_]*(\d+)[\W_]*(tail|head)", re.IGNORECASE)


def load_model(finetuned=False, adapter_dir=None, use_4bit=True, device="auto"):
    resolved_device = select_best_cuda_device() if device == "auto" else device

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["visual"],
        )

    device_map = {"": int(resolved_device.split(":")[1])} if use_4bit and resolved_device.startswith("cuda:") else resolved_device

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=DTYPE if resolved_device.startswith("cuda") else torch.float32,
    )
    if not use_4bit:
        model = model.to(resolved_device)

    if finetuned:
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()
    return model, processor


def _extract_json_array(response):
    text = response.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _coerce_point(raw):
    if not isinstance(raw, (list, tuple)):
        return None
    flat = []
    for part in raw:
        if isinstance(part, str) and "," in part:
            flat.extend(part.split(","))
        else:
            flat.append(part)
    if len(flat) != 2:
        return None
    try:
        return float(str(flat[0]).strip()), float(str(flat[1]).strip())
    except (TypeError, ValueError):
        return None


def _parse_arrows(response):
    by_index = defaultdict(dict)
    order = []
    for item in _extract_json_array(response):
        if not isinstance(item, dict):
            continue
        point = _coerce_point(item.get("point_2d"))
        if point is None:
            continue
        match = _LABEL_RE.search(str(item.get("label", "")))
        if match is None:
            continue
        index, role = match.group(1), match.group(2).lower()
        if index not in by_index:
            order.append(index)
        by_index[index][role] = point

    arrows = []
    for index in order:
        ends = by_index[index]
        if "tail" in ends and "head" in ends:
            arrows.append({"tail": list(ends["tail"]), "head": list(ends["head"])})
    return arrows


def resize_for_model(image, max_image_size=1792):
    orig_w, orig_h = image.size

    grid_cap = round(max_image_size / GRID) * GRID
    target_w = min(round(orig_w / GRID) * GRID, grid_cap) or GRID
    target_h = min(round(orig_h / GRID) * GRID, grid_cap) or GRID
    if max(orig_w, orig_h) > grid_cap:
        shrink = grid_cap / max(orig_w, orig_h)
        target_w = round(orig_w * shrink / GRID) * GRID or GRID
        target_h = round(orig_h * shrink / GRID) * GRID or GRID

    scale_x, scale_y = target_w / orig_w, target_h / orig_h
    if (target_w, target_h) != (orig_w, orig_h):
        image = image.resize((target_w, target_h), Image.LANCZOS)
    return image, scale_x, scale_y


def single_image(
    model, processor, image_path, prompt=None, max_image_size=1792, max_new_tokens=3072, normalized=False
):
    prompt = prompt or (PROMPT_NORMALIZED if normalized else PROMPT)
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    image, scale_x, scale_y = resize_for_model(image, max_image_size)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    pixel_count = image.width * image.height
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True,
        min_pixels=pixel_count, max_pixels=pixel_count,
    )
    input_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(input_device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    arrows = _parse_arrows(response)
    for arrow in arrows:
        if normalized:
            arrow["tail"] = [round(arrow["tail"][0] / NORM_SCALE * orig_w, 1), round(arrow["tail"][1] / NORM_SCALE * orig_h, 1)]
            arrow["head"] = [round(arrow["head"][0] / NORM_SCALE * orig_w, 1), round(arrow["head"][1] / NORM_SCALE * orig_h, 1)]
        else:
            arrow["tail"] = [round(arrow["tail"][0] / scale_x, 1), round(arrow["tail"][1] / scale_y, 1)]
            arrow["head"] = [round(arrow["head"][0] / scale_x, 1), round(arrow["head"][1] / scale_y, 1)]
    return arrows


def full_dataset(
    model, processor, input_dir, output_dir, prompt=None, max_image_size=1792, max_new_tokens=3072, normalized=False
):
    input_path, output_path = Path(input_dir), Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(set(
        p for ext in IMAGE_EXTENSIONS for p in list(input_path.glob(f"*{ext}")) + list(input_path.glob(f"*{ext.upper()}"))
    ))

    results = {}
    for img_path in tqdm(image_files, desc="Qwen3-VL inference"):
        arrows = single_image(
            model, processor, str(img_path), prompt, max_image_size, max_new_tokens, normalized=normalized
        )
        results[img_path.stem] = arrows
        with open(output_path / f"{img_path.stem}.json", "w") as f:
            json.dump(arrows, f, indent=2)
    return results


def zero_shot(input_dir, output_dir, prompt=None, use_4bit=True, device="auto", **kwargs):
    model, processor = load_model(finetuned=False, use_4bit=use_4bit, device=device)
    return full_dataset(model, processor, input_dir, output_dir, prompt, **kwargs)


def finetuned(input_dir, output_dir, adapter_dir, prompt=None, use_4bit=True, device="auto", **kwargs):
    model, processor = load_model(finetuned=True, adapter_dir=adapter_dir, use_4bit=use_4bit, device=device)
    return full_dataset(model, processor, input_dir, output_dir, prompt, **kwargs)


def convert_to_bbox(tail, head, img_w=None, img_h=None):
    x1, y1 = min(tail[0], head[0]), min(tail[1], head[1])
    x2, y2 = max(tail[0], head[0]), max(tail[1], head[1])
    if img_w is not None:
        x1, x2 = max(0, min(x1, img_w - 1)), max(0, min(x2, img_w - 1))
    if img_h is not None:
        y1, y2 = max(0, min(y1, img_h - 1)), max(0, min(y2, img_h - 1))
    return x1, y1, x2, y2
