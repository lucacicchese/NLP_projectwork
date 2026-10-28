import argparse
import gc
import json
import math
import random
import shutil
from pathlib import Path

import torch
from ntloss import NTLoss
from peft import PeftModel, prepare_model_for_kbit_training
from PIL import Image
from torch.optim import AdamW
from tqdm import tqdm

import box_check
import metrics
from finetune import NTL_WEIGHT
from inference import IMAGE_EXTENSIONS, NORM_SCALE, PROMPT_NORMALIZED, parse_arrows, load_model, resize_for_model

TOKENS_PER_ARROW = 80
MAX_NEW_TOKENS = 3072
ROW_BAND = 50


def load_examples(dataset_dir):
    dataset_dir = Path(dataset_dir)
    examples = []
    for image_path in sorted((dataset_dir / "images").iterdir()):
        labels = dataset_dir / "labels" / f"{image_path.stem}.csv"
        if image_path.suffix.lower() in IMAGE_EXTENSIONS and labels.exists():
            gt_arrows = box_check.load_gt_arrows(labels)
            if gt_arrows:
                examples.append((image_path, gt_arrows))
    return examples


def chat_prompt(processor):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT_NORMALIZED}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def model_inputs(processor, images, texts, device):
    pixels = images[0].width * images[0].height
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, min_pixels=pixels, max_pixels=pixels)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def sample_union(model, processor, image, n_gt, args):
    orig_w, orig_h = image.size
    resized, _, _ = resize_for_model(image, args.max_image_size)
    device = model.get_input_embeddings().weight.device
    inputs = model_inputs(processor, [resized] * args.n_samples, [chat_prompt(processor)] * args.n_samples, device)
    prompt_len = inputs["input_ids"].shape[1]
    max_new_tokens = min(MAX_NEW_TOKENS, TOKENS_PER_ARROW * (n_gt + 5))
    model.eval()
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=0.9, repetition_penalty=1.1,
            )
    finally:
        model.train()
    texts = processor.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    arrows = []
    for text in texts:
        for a in parse_arrows(text):
            arrows.append({
                "tail": [a["tail"][0] / NORM_SCALE * orig_w, a["tail"][1] / NORM_SCALE * orig_h],
                "head": [a["head"][0] / NORM_SCALE * orig_w, a["head"][1] / NORM_SCALE * orig_h],
            })
    return arrows


def build_target(gt_arrows, candidates, img_w, img_h, margin, threshold, tol_frac):
    tol = tol_frac * math.hypot(img_w, img_h)
    boxes = [box_check.arrow_box(t, h, img_w, img_h, margin) for t, h in gt_arrows]
    groups = [[] for _ in gt_arrows]
    for c in candidates:
        best = None
        for i, ((t, h), box) in enumerate(zip(gt_arrows, boxes)):
            dt, dh = math.dist(c["tail"], t), math.dist(c["head"], h)
            if dt <= tol and dh <= tol and box_check.fraction_inside(c["tail"], c["head"], box) >= threshold:
                if best is None or dt + dh < best[0]:
                    best = (dt + dh, i)
        if best is not None:
            groups[best[1]].append(c)

    arrows, from_model = [], 0
    for (t, h), group in zip(gt_arrows, groups):
        if group:
            from_model += 1
            arrows.append({
                "tail": [sum(c["tail"][0] for c in group) / len(group), sum(c["tail"][1] for c in group) / len(group)],
                "head": [sum(c["head"][0] for c in group) / len(group), sum(c["head"][1] for c in group) / len(group)],
            })
        else:
            arrows.append({"tail": list(t), "head": list(h)})
    return arrows, from_model


def raster_order(arrows, img_w, img_h, band=ROW_BAND):
    def key(arrow):
        x, y = arrow["head"]
        return int(y / img_h * NORM_SCALE // band), x / img_w
    return sorted(arrows, key=key)


def target_text(arrows, img_w, img_h):
    entries = []
    for i, arrow in enumerate(raster_order(arrows, img_w, img_h), start=1):
        for role in ("tail", "head"):
            x, y = arrow[role]
            point = [round(x / img_w * NORM_SCALE, 1), round(y / img_h * NORM_SCALE, 1)]
            entries.append({"point_2d": point, "label": f"arrow_{i}_{role}"})
    return json.dumps(entries)


def build_round(model, processor, examples, cache_path, args):
    if cache_path.exists():
        candidates = json.loads(cache_path.read_text())
    else:
        candidates = {}
        for image_path, gt_arrows in tqdm(examples, desc=f"sampling -> {cache_path.name}"):
            image = Image.open(image_path).convert("RGB")
            candidates[image_path.name] = sample_union(model, processor, image, len(gt_arrows), args)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        cache_path.write_text(json.dumps(candidates))

    targets = []
    stats = {"images": len(examples), "gt": 0, "candidates": 0, "from_model": 0, "from_gt": 0}
    for image_path, gt_arrows in examples:
        with Image.open(image_path) as im:
            w, h = im.size
        found = candidates.get(image_path.name, [])
        arrows, from_model = build_target(gt_arrows, found, w, h, args.margin, args.threshold, args.tol_frac)
        targets.append((image_path, target_text(arrows, w, h)))
        stats["gt"] += len(gt_arrows)
        stats["candidates"] += len(found)
        stats["from_model"] += from_model
        stats["from_gt"] += len(gt_arrows) - from_model
    return targets, stats


def train_epoch(model, processor, ntl_fn, optimizer, targets, args, desc):
    prompt = chat_prompt(processor)
    device = model.get_input_embeddings().weight.device
    total, steps = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(targets, desc=desc)
    for k, (image_path, text) in enumerate(progress, start=1):
        image, _, _ = resize_for_model(Image.open(image_path).convert("RGB"), args.max_image_size)
        prompt_len = model_inputs(processor, [image], [prompt], "cpu")["input_ids"].shape[1]
        inputs = model_inputs(processor, [image], [prompt + text], device)
        labels = inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        try:
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss + args.ntl_weight * ntl_fn(outputs.logits[:, :-1, :], labels[:, 1:])
            value = loss.item()
            if math.isfinite(value):
                (loss / args.grad_accum).backward()
                total += value
                steps += 1
                progress.set_postfix(loss=f"{value:.4f}", avg=f"{total / steps:.4f}")
        except torch.OutOfMemoryError:
            tqdm.write(f"Skipping {image_path.name}: CUDA out of memory")
            optimizer.zero_grad(set_to_none=True)
        outputs = loss = None
        del inputs, labels
        if k % args.grad_accum == 0 or k == len(targets):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return total / steps if steps else None


def build_parser():
    parser = argparse.ArgumentParser(description="Training on sampled arrows kept by the GT boxes")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--init_adapter", default="adapter_normalized_v4")
    parser.add_argument("--adapter_dir", default="adapter_unifi_box")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--epochs_per_round", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ntl_weight", type=float, default=NTL_WEIGHT)
    parser.add_argument("--margin", type=float, default=box_check.MARGIN_FRAC)
    parser.add_argument("--threshold", type=float, default=box_check.INSIDE_THRESHOLD)
    parser.add_argument("--tol_frac", type=float, default=metrics.PCK_TOLERANCE_FRAC)
    parser.add_argument("--max_image_size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    adapter_dir = Path(args.adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        shutil.copytree(args.init_adapter, adapter_dir, dirs_exist_ok=True)
        (adapter_dir / "progress.json").unlink(missing_ok=True)

    model, processor = load_model(finetuned=False, device=args.device)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    model.train()
    ntl_fn = NTLoss(tokenizer=processor.tokenizer, vocab_size=model.get_output_embeddings().out_features)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    examples = load_examples(args.dataset)
    progress_path = adapter_dir / "box_progress.json"
    done = json.loads(progress_path.read_text()) if progress_path.exists() else {"round": 0, "epoch": 0}

    for r in range(args.rounds):
        todo = [e for e in range(args.epochs_per_round) if (r, e) >= (done["round"], done["epoch"])]
        if not todo:
            continue
        targets, stats = build_round(model, processor, examples, adapter_dir / f"candidates_round{r + 1}.json", args)
        tqdm.write(json.dumps({"round": r + 1, **stats}))
        for e in todo:
            order = list(targets)
            random.Random(args.seed + 1000 * r + e).shuffle(order)
            loss = train_epoch(model, processor, ntl_fn, optimizer, order, args, f"round {r + 1} epoch {e + 1}/{args.epochs_per_round}")
            log = {"round": r + 1, "epoch": e + 1, "loss": loss, **stats}
            tqdm.write(json.dumps(log))
            with open(adapter_dir / "box_train_log.jsonl", "a") as f:
                f.write(json.dumps(log) + "\n")
            model.save_pretrained(adapter_dir)
            progress_path.write_text(json.dumps({"round": r, "epoch": e + 1}))


if __name__ == "__main__":
    main()
