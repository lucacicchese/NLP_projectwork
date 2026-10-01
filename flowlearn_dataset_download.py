import argparse
import csv
import io
import random
from pathlib import Path

import requests
from PIL import Image

BASE_URL = "https://huggingface.co/datasets/jopan/FlowLearn/resolve/main/SimFlowchart/images"
VARIANT_DIRS = {"word": "mermaid_word", "char": "mermaid_char_v2"}
TOTAL_AVAILABLE = 10000
OUTPUT_DIR = Path("training_data/train")


def fetch_json(base_url, idx):
    response = requests.get(f"{base_url}/parsed/{idx}.json", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_image(base_url, idx):
    response = requests.get(f"{base_url}/jpeg/{idx}.jpeg", timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def convert_links(parsed):
    rows = []
    for link in parsed.get("links", {}).values():
        for bbox, label in ((link["start_point_bbox"], "arrow_start"), (link["end_point_bbox"], "arrow_end"), (link["link_bbox"], "arrow")):
            x0, y0, x1, y1 = bbox
            rows.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), label))
    return rows


def download(variant="word", count=2000, seed=0):
    base_url = f"{BASE_URL}/{VARIANT_DIRS[variant]}"
    images_dir = OUTPUT_DIR / "images"
    labels_dir = OUTPUT_DIR / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    ids = list(range(TOTAL_AVAILABLE))
    random.Random(seed).shuffle(ids)
    ids = ids[:count]

    saved = 0
    for i, idx in enumerate(ids, start=1):
        try:
            parsed = fetch_json(base_url, idx)
            rows = convert_links(parsed)
            if not rows:
                continue
            image = fetch_image(base_url, idx)
        except Exception as e:
            print(f"[{i}/{len(ids)}] Skipping {idx}: {e}")
            continue

        stem = f"flowlearn_{variant}_{idx}"
        image.save(images_dir / f"{stem}.jpg", format="JPEG")
        with open(labels_dir / f"{stem}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["xmin", "ymin", "xmax", "ymax", "label"])
            writer.writerows(rows)

        saved += 1
        print(f"[{i}/{len(ids)}] Saved {stem} ({len(rows) // 3} arrows)")

    print(f"\nDone. Saved {saved}/{len(ids)} flowcharts to {images_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Download and convert FlowLearn SimFlowchart into our training_data format")
    parser.add_argument("--variant", choices=list(VARIANT_DIRS), default="word", help="Text style of the synthetic flowcharts")
    parser.add_argument("--count", type=int, default=2000, help=f"How many flowcharts to download (max {TOTAL_AVAILABLE})")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for picking which flowcharts to download")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    download(variant=args.variant, count=args.count, seed=args.seed)
