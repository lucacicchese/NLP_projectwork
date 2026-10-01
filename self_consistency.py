import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

import metrics
import visualize
from inference import IMAGE_EXTENSIONS, NORM_SCALE, PROMPT, PROMPT_NORMALIZED, _parse_arrows, load_model, resize_for_model


def sample_arrows(model, processor, image_path, prompt, max_image_size, normalized, n_samples, temperature, max_new_tokens=3072):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    resized, scale_x, scale_y = resize_for_model(image, max_image_size)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pixel_count = resized.width * resized.height
    inputs = processor(
        text=[text], images=[resized], return_tensors="pt", padding=True,
        min_pixels=pixel_count, max_pixels=pixel_count,
    )
    input_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(input_device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    samples = []
    for _ in range(n_samples):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=0.9, repetition_penalty=1.1,
            )
        trimmed = output_ids[0][prompt_len:]
        response = processor.batch_decode([trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        arrows = _parse_arrows(response)
        for arrow in arrows:
            if normalized:
                arrow["tail"] = [arrow["tail"][0] / NORM_SCALE * orig_w, arrow["tail"][1] / NORM_SCALE * orig_h]
                arrow["head"] = [arrow["head"][0] / NORM_SCALE * orig_w, arrow["head"][1] / NORM_SCALE * orig_h]
            else:
                arrow["tail"] = [arrow["tail"][0] / scale_x, arrow["tail"][1] / scale_y]
                arrow["head"] = [arrow["head"][0] / scale_x, arrow["head"][1] / scale_y]
        samples.append(arrows)
    return samples


def aggregate(samples, tol):
    clusters = []
    for sample_idx, arrows in enumerate(samples):
        for arrow in arrows:
            tail, head = tuple(arrow["tail"]), tuple(arrow["head"])
            match = None
            for c in clusters:
                ct = (sum(p[0] for p in c["tails"]) / len(c["tails"]), sum(p[1] for p in c["tails"]) / len(c["tails"]))
                ch = (sum(p[0] for p in c["heads"]) / len(c["heads"]), sum(p[1] for p in c["heads"]) / len(c["heads"]))
                if math.hypot(tail[0] - ct[0], tail[1] - ct[1]) <= tol and math.hypot(head[0] - ch[0], head[1] - ch[1]) <= tol:
                    match = c
                    break
            if match is None:
                match = {"tails": [], "heads": [], "samples": set()}
                clusters.append(match)
            match["tails"].append(tail)
            match["heads"].append(head)
            match["samples"].add(sample_idx)
    return clusters


def consensus_arrows(clusters, n_samples, min_frac):
    min_support = max(1, round(min_frac * n_samples))
    arrows = []
    for c in clusters:
        if len(c["samples"]) < min_support:
            continue
        tx = round(sum(p[0] for p in c["tails"]) / len(c["tails"]), 1)
        ty = round(sum(p[1] for p in c["tails"]) / len(c["tails"]), 1)
        hx = round(sum(p[0] for p in c["heads"]) / len(c["heads"]), 1)
        hy = round(sum(p[1] for p in c["heads"]) / len(c["heads"]), 1)
        arrows.append({"tail": [tx, ty], "head": [hx, hy]})
    return arrows


def run(model, processor, images_dir, labels_dir, output_dir, prompt, max_image_size, normalized, n_samples, temperature, min_frac, tol_frac, show):
    images_dir, labels_dir, output_dir = Path(images_dir), Path(labels_dir), Path(output_dir)
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(set(
        p for ext in IMAGE_EXTENSIONS for p in list(images_dir.glob(f"*{ext}")) + list(images_dir.glob(f"*{ext.upper()}"))
    ))

    predictions, image_sizes = {}, {}
    for img_path in tqdm(image_files, desc="Self-consistency inference"):
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        samples = sample_arrows(model, processor, img_path, prompt, max_image_size, normalized, n_samples, temperature)
        clusters = aggregate(samples, tol_frac * math.hypot(w, h))
        arrows = consensus_arrows(clusters, n_samples, min_frac)

        predictions[img_path.stem] = arrows
        image_sizes[img_path.stem] = (w, h)
        with open(json_dir / f"{img_path.stem}.json", "w") as f:
            json.dump(arrows, f, indent=2)

    if labels_dir.exists():
        report = metrics.evaluate(predictions, labels_dir, image_sizes)
        print(report)

    if show:
        imgs_dir = output_dir / "imgs"
        imgs_dir.mkdir(parents=True, exist_ok=True)
        for stem, arrows in predictions.items():
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    gt_boxes = metrics.load_gt_boxes(labels_dir / f"{stem}.csv") if labels_dir.exists() else None
                    visualize.draw_boxes(str(candidate), arrows, gt_boxes, str(imgs_dir / f"{stem}_annotated{ext}"))
                    break


def build_parser():
    parser = argparse.ArgumentParser(description="Self-consistency sampling for arrow detection")
    parser.add_argument("--dataset", default="training_data/val", help="Dataset split dir with images/ and labels/ subdirs")
    parser.add_argument("--adapter_dir", default="adapter_normalized_v4", help="LoRA adapter directory")
    parser.add_argument("--output_dir", default="results/self_consistency", help="Where to save predictions and visualizations")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of sampled generations per image")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min_frac", type=float, default=0.5, help="Fraction of samples an arrow must appear in to be kept")
    parser.add_argument("--tol_frac", type=float, default=0.03, help="Clustering tolerance, fraction of image diagonal")
    parser.add_argument("--max_image_size", type=int, default=384)
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--print", "--show", dest="show", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    model, processor = load_model(finetuned=True, adapter_dir=args.adapter_dir, use_4bit=args.use_4bit, device=args.device)
    prompt = PROMPT_NORMALIZED if args.normalized else PROMPT
    dataset_dir = Path(args.dataset)
    run(
        model, processor, dataset_dir / "images", dataset_dir / "labels", args.output_dir, prompt,
        args.max_image_size, args.normalized, args.n_samples, args.temperature, args.min_frac, args.tol_frac, args.show,
    )


if __name__ == "__main__":
    main()
