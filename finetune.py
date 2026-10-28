import csv
import gc
import json
import math
from pathlib import Path

import bitsandbytes as bnb
import torch
from ntloss import NTLoss
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.optim import AdamW
from tqdm import tqdm

import box_check
from inference import IMAGE_EXTENSIONS, NORM_SCALE, PROMPT, PROMPT_NORMALIZED, load_model, resize_for_model, single_image

ARROW_LABELS = {"arrow", "Arrow", "ARROW", "6"}
NTL_WEIGHT = 0.3
CHECKPOINT_EVERY_STEPS = 200

LORA_TARGET_MODULES = r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$|.*merger\.(linear_fc1|linear_fc2)$"
LORA_RANK = 48


def lora_config(r=LORA_RANK):
    return LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none", target_modules=LORA_TARGET_MODULES)


def center(row):
    return (float(row["xmin"]) + float(row["xmax"])) / 2, (float(row["ymin"]) + float(row["ymax"])) / 2


def contains(box_row, point, tol=5.0):
    x, y = point
    return (
        float(box_row["xmin"]) - tol <= x <= float(box_row["xmax"]) + tol
        and float(box_row["ymin"]) - tol <= y <= float(box_row["ymax"]) + tol
    )


def target_text(labels_csv, scale_x, scale_y, orig_w, orig_h, normalized=False):
    with open(labels_csv) as f:
        rows = list(csv.DictReader(f))

    start_rows = [r for r in rows if (r.get("label") or "").strip() == "arrow_start"]
    end_rows = [r for r in rows if (r.get("label") or "").strip() == "arrow_end"]

    arrows = []
    for row in rows:
        label = (row.get("label") or "").strip()
        if label and label not in ARROW_LABELS:
            continue
        try:
            x1, y1, x2, y2 = (float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"]))
        except (KeyError, TypeError, ValueError):
            continue

        starts_in = [s for s in start_rows if contains(row, center(s))]
        ends_in = [e for e in end_rows if contains(row, center(e))]
        if len(starts_in) == 1 and len(ends_in) == 1:
            x1, y1 = center(starts_in[0])
            x2, y2 = center(ends_in[0])

        if normalized:
            arrows.append((
                x1 / orig_w * NORM_SCALE, y1 / orig_h * NORM_SCALE,
                x2 / orig_w * NORM_SCALE, y2 / orig_h * NORM_SCALE,
            ))
        else:
            arrows.append((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y))

    entries = []
    for i, (x1, y1, x2, y2) in enumerate(arrows, start=1):
        entries.append({"point_2d": [round(x1, 1), round(y1, 1)], "label": f"arrow_{i}_tail"})
        entries.append({"point_2d": [round(x2, 1), round(y2, 1)], "label": f"arrow_{i}_head"})
    return json.dumps(entries)


def build_examples(images_dir, labels_dir, max_image_size=1792, normalized=False):
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    examples = []
    for csv_path in sorted(labels_dir.glob("*.csv")):
        for ext in (".jpg", ".jpeg", ".png"):
            image_path = images_dir / f"{csv_path.stem}{ext}"
            if image_path.exists():
                image = Image.open(image_path).convert("RGB")
                orig_w, orig_h = image.size
                _, scale_x, scale_y = resize_for_model(image, max_image_size)
                text = target_text(csv_path, scale_x, scale_y, orig_w, orig_h, normalized)
                examples.append((image_path, text))
                break
    return examples


def is_synthetic(image_path):
    return image_path.name.startswith("flowlearn_")


def progress_path(adapter_dir):
    return Path(adapter_dir) / "progress.json"


def load_progress(adapter_dir):
    p = progress_path(adapter_dir)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_checkpoint(model, adapter_dir, phase, epoch):
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    progress_path(adapter_dir).write_text(json.dumps({"phase": phase, "epoch": epoch}))


def box_validation(model, processor, val_images_dir, val_labels_dir, max_images, prompt, max_image_size, normalized):
    image_paths = sorted(p for p in Path(val_images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    image_paths = [p for p in image_paths if (Path(val_labels_dir) / f"{p.stem}.csv").exists()]
    if max_images:
        image_paths = image_paths[:max_images]
    model.eval()
    predictions, sizes = {}, {}
    try:
        for image_path in tqdm(image_paths, desc="Box validation", leave=False):
            predictions[image_path.stem] = single_image(
                model, processor, str(image_path), prompt, max_image_size, normalized=normalized
            )
            with Image.open(image_path) as im:
                sizes[image_path.stem] = im.size
    finally:
        model.train()
    return predictions, sizes


def make_box_callback(model, processor, box_val_dir, box_val_max, margin_start, margin_end, threshold, prompt, max_image_size, normalized, adapter_dir):
    val_dir = Path(box_val_dir)
    log_path = Path(adapter_dir) / "box_val.jsonl"
    best = {"f1": -1.0}

    def callback(phase, epoch, epochs):
        predictions, sizes = box_validation(
            model, processor, val_dir / "images", val_dir / "labels", box_val_max, prompt, max_image_size, normalized
        )
        margin = box_check.margin_for_epoch(epoch, epochs, margin_start, margin_end)
        current = box_check.evaluate_box(predictions, val_dir / "labels", sizes, margin, threshold)
        final = box_check.evaluate_box(predictions, val_dir / "labels", sizes, margin_end, threshold)
        tqdm.write(
            f"Phase {phase} epoch {epoch + 1}/{epochs} box val: F1 {current['f1']:.3f} (margin {margin:.3f}), "
            f"F1 {final['f1']:.3f} (margin {margin_end:.3f})"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({"phase": phase, "epoch": epoch + 1, "current": current, "final_margin": final}) + "\n")
        if final["f1"] > best["f1"]:
            best.update(final, phase=phase, epoch=epoch + 1)
            model.save_pretrained(Path(adapter_dir) / "best_box")
            (Path(adapter_dir) / "best_box" / "box_val.json").write_text(json.dumps(best, indent=2))

    return callback


def run_phase(model, processor, ntl_fn, ntl_weight, examples, epochs, lr, max_image_size, prompt, adapter_dir, phase, start_epoch, on_epoch_end=None, low_mem=False):
    if low_mem:
        optimizer = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=lr)
    else:
        optimizer = AdamW(model.parameters(), lr=lr)
    input_device = model.get_input_embeddings().weight.device
    step = 0

    for epoch in range(start_epoch, epochs):
        progress = tqdm(examples, desc=f"Phase {phase} epoch {epoch + 1}/{epochs}")
        epoch_loss, n_steps = 0.0, 0
        for image_path, target_text in progress:
            image = Image.open(image_path).convert("RGB")
            image, _, _ = resize_for_model(image, max_image_size)
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_text = prompt_text + target_text

            pixel_count = image.width * image.height
            inputs = processor(
                text=[full_text], images=[image], return_tensors="pt", padding=True,
                min_pixels=pixel_count, max_pixels=pixel_count,
            )
            prompt_inputs = processor(
                text=[prompt_text], images=[image], return_tensors="pt", padding=True,
                min_pixels=pixel_count, max_pixels=pixel_count,
            )
            inputs = {k: v.to(input_device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            labels = inputs["input_ids"].clone()
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels[:, :prompt_len] = -100

            try:
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                if ntl_weight > 0:
                    loss = loss + ntl_weight * ntl_fn(outputs.logits[:, :-1, :], labels[:, 1:])
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    tqdm.write(f"Skipping {image_path.name}: non-finite loss ({loss_value})")
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss_value
                    n_steps += 1
                    step += 1
                    progress.set_postfix(loss=f"{loss_value:.4f}", avg=f"{epoch_loss / n_steps:.4f}")
                    if step % CHECKPOINT_EVERY_STEPS == 0:
                        save_checkpoint(model, adapter_dir, phase, epoch)
            except torch.OutOfMemoryError:
                tqdm.write(f"Skipping {image_path.name}: CUDA out of memory")
                outputs = None

            optimizer.zero_grad(set_to_none=True)
            del inputs, prompt_inputs, labels, outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if n_steps:
            tqdm.write(f"Phase {phase} epoch {epoch + 1}/{epochs} mean loss: {epoch_loss / n_steps:.4f} ({n_steps} steps)")
        save_checkpoint(model, adapter_dir, phase, epoch + 1)
        if on_epoch_end is not None:
            on_epoch_end(phase, epoch, epochs)


def finetune(
    train_images_dir, train_labels_dir, adapter_dir, epochs=1, lr=1e-4, use_4bit=True, device="auto",
    max_image_size=384, normalized=False, prompt=None, real_epochs=1, real_lr=2e-5, ntl_weight=NTL_WEIGHT,
    box_val_dir=None, box_val_max=0, box_margin_start=box_check.MARGIN_FRAC, box_margin_end=box_check.MARGIN_FRAC,
    box_threshold=box_check.INSIDE_THRESHOLD, low_mem=False, lora_r=LORA_RANK,
):
    model, processor = load_model(finetuned=False, use_4bit=use_4bit, device=device)
    prompt = prompt or (PROMPT_NORMALIZED if normalized else PROMPT)
    checkpointing_kwargs = {"use_reentrant": False}
    if use_4bit and not low_mem:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=checkpointing_kwargs
        )
    else:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpointing_kwargs)
        model.enable_input_require_grads()

    if (Path(adapter_dir) / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    else:
        model = get_peft_model(model, lora_config(lora_r))
    if low_mem:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.bfloat16)
    model.train()
    ntl_fn = NTLoss(tokenizer=processor.tokenizer, vocab_size=model.get_output_embeddings().out_features)

    examples = build_examples(train_images_dir, train_labels_dir, max_image_size, normalized)
    synthetic = [e for e in examples if is_synthetic(e[0])]
    real = [e for e in examples if not is_synthetic(e[0])]

    phases = []
    if synthetic:
        phases.append(("synthetic", synthetic, epochs, lr))
    phases.append(("real", real or examples, real_epochs, real_lr))

    on_epoch_end = None
    if box_val_dir:
        on_epoch_end = make_box_callback(
            model, processor, box_val_dir, box_val_max, box_margin_start, box_margin_end, box_threshold,
            prompt, max_image_size, normalized, adapter_dir,
        )

    progress = load_progress(adapter_dir)
    resumed = False
    for phase, phase_examples, phase_epochs, phase_lr in phases:
        if not resumed and progress.get("phase") and progress["phase"] != phase:
            continue
        start_epoch = progress["epoch"] if (not resumed and progress.get("phase") == phase) else 0
        resumed = True
        run_phase(model, processor, ntl_fn, ntl_weight, phase_examples, phase_epochs, phase_lr, max_image_size, prompt, adapter_dir, phase, start_epoch, on_epoch_end, low_mem)

    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    return adapter_dir
