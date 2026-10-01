"""Draw arrow predictions from a JSON file onto an image, for quick visual debugging.

Usage:
    python draw_json.py path/to/predictions.json
    python draw_json.py path/to/predictions.json --image flowchart-example037 --output out.jpg
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

PRED_COLOR = "#00C853"
GT_COLOR = "#2196F3"
ARROW_LABELS = {"arrow", "Arrow", "ARROW", "6"}
NORM_SCALE = 1000  # must match inference.NORM_SCALE / PROMPT_NORMALIZED

IMAGES_DIR = Path("training_data/val/images")
LABELS_DIR = Path("training_data/val/labels")

_LABEL_RE = re.compile(r"arrow[\W_]*(\d+)[\W_]*(tail|head)", re.IGNORECASE)


def normalize_arrows(raw):
    """Accept either [{tail, head}, ...] or the raw model format
    [{"point_2d": [x, y], "label": "arrow_1_tail"}, ...]."""
    if raw and isinstance(raw[0], dict) and "tail" in raw[0]:
        return raw

    by_index = defaultdict(dict)
    order = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        point = item.get("point_2d")
        if not (isinstance(point, (list, tuple)) and len(point) == 2):
            continue
        match = _LABEL_RE.search(str(item.get("label", "")))
        if match is None:
            continue
        index, role = match.group(1), match.group(2).lower()
        if index not in by_index:
            order.append(index)
        by_index[index][role] = list(point)

    arrows = []
    for index in order:
        ends = by_index[index]
        if "tail" in ends and "head" in ends:
            arrows.append({"tail": ends["tail"], "head": ends["head"]})
    return arrows


def denormalize_arrows(arrows, w, h):
    """Map tail/head points from a 0-1000 normalized scale to actual pixel coordinates."""
    out = []
    for arrow in arrows:
        tail = [arrow["tail"][0] / NORM_SCALE * w, arrow["tail"][1] / NORM_SCALE * h]
        head = [arrow["head"][0] / NORM_SCALE * w, arrow["head"][1] / NORM_SCALE * h]
        out.append({"tail": tail, "head": head})
    return out


def load_gt_boxes(labels_csv):
    labels_csv = Path(labels_csv)
    boxes = []
    if not labels_csv.exists():
        return boxes
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip()
            if label and label not in ARROW_LABELS:
                continue
            try:
                boxes.append((float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])))
            except (KeyError, TypeError, ValueError):
                continue
    return boxes


def convert_to_bbox(tail, head, img_w=None, img_h=None):
    x1, y1 = min(tail[0], head[0]), min(tail[1], head[1])
    x2, y2 = max(tail[0], head[0]), max(tail[1], head[1])
    if img_w is not None:
        x1, x2 = max(0, min(x1, img_w - 1)), max(0, min(x2, img_w - 1))
    if img_h is not None:
        y1, y2 = max(0, min(y1, img_h - 1)), max(0, min(y2, img_h - 1))
    return x1, y1, x2, y2


def draw_raw(json_path, image_stem, output_path, normalized=False):
    """Draw straight from the raw model format: a line per arrow index from its
    *_tail point to its *_head point, with each point's label written next to it.
    Bypasses normalize_arrows()/convert_to_bbox() entirely, for debugging whether
    those introduce the offset."""
    with open(json_path) as f:
        raw = json.load(f)

    image_path = find_image(image_stem)
    if image_path is None:
        raise FileNotFoundError(f"No image found for stem '{image_stem}' in {IMAGES_DIR}")

    img = Image.open(image_path).convert("RGB")
    draw_ctx = ImageDraw.Draw(img)
    w, h = img.size

    gt_boxes = load_gt_boxes(LABELS_DIR / f"{image_stem}.csv")
    for x1, y1, x2, y2 in gt_boxes:
        draw_ctx.rectangle([(x1, y1), (x2, y2)], outline=GT_COLOR, width=3)

    points_by_index = defaultdict(dict)
    for item in raw:
        if not isinstance(item, dict):
            continue
        point = item.get("point_2d")
        if not (isinstance(point, (list, tuple)) and len(point) == 2):
            continue
        if normalized:
            point = (point[0] / NORM_SCALE * w, point[1] / NORM_SCALE * h)
        label = str(item.get("label", ""))
        match = _LABEL_RE.search(label)
        if match is None:
            continue
        index, role = match.group(1), match.group(2).lower()
        points_by_index[index][role] = (tuple(point), label)

    for index, ends in points_by_index.items():
        if "tail" in ends and "head" in ends:
            (tail_xy, _), (head_xy, _) = ends["tail"], ends["head"]
            draw_ctx.line([tail_xy, head_xy], fill=PRED_COLOR, width=3)
        for xy, label in ends.values():
            r = 4
            draw_ctx.ellipse([xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r], fill=PRED_COLOR)
            draw_ctx.text((xy[0] + r + 2, xy[1] - r), label, fill=PRED_COLOR)

    draw_ctx.text((5, 5), "ground truth", fill=GT_COLOR)
    draw_ctx.text((5, 19), "prediction (raw)", fill=PRED_COLOR)

    img.save(output_path)
    n = sum(1 for ends in points_by_index.values() if "tail" in ends and "head" in ends)
    print(f"{n} arrows drawn -> {output_path}")


def find_image(stem):
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = IMAGES_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def draw(json_path, image_stem, output_path, normalized=False):
    with open(json_path) as f:
        arrows = normalize_arrows(json.load(f))

    image_path = find_image(image_stem)
    if image_path is None:
        raise FileNotFoundError(f"No image found for stem '{image_stem}' in {IMAGES_DIR}")

    img = Image.open(image_path).convert("RGB")
    draw_ctx = ImageDraw.Draw(img)
    w, h = img.size

    if normalized:
        arrows = denormalize_arrows(arrows, w, h)

    gt_boxes = load_gt_boxes(LABELS_DIR / f"{image_stem}.csv")
    for x1, y1, x2, y2 in gt_boxes:
        draw_ctx.rectangle([(x1, y1), (x2, y2)], outline=GT_COLOR, width=3)

    for arrow in arrows:
        x1, y1, x2, y2 = convert_to_bbox(arrow["tail"], arrow["head"], w, h)
        draw_ctx.rectangle([(x1, y1), (x2, y2)], outline=PRED_COLOR, width=3)
        draw_ctx.line([tuple(arrow["tail"]), tuple(arrow["head"])], fill=PRED_COLOR, width=3)

    draw_ctx.text((5, 5), "ground truth", fill=GT_COLOR)
    draw_ctx.text((5, 19), "prediction", fill=PRED_COLOR)

    img.save(output_path)
    print(f"{len(arrows)} arrows drawn -> {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Draw arrow predictions from a JSON file on an image")
    parser.add_argument("json_path", help="Path to a predictions JSON file (list of {tail, head})")
    parser.add_argument("--image", default="flowchart-example007", help="Image stem to draw on (default: flowchart-example007)")
    parser.add_argument("--output", default=None, help="Output path (default: <json_stem>_annotated.jpg)")
    parser.add_argument(
        "--raw", action="store_true",
        help="Draw directly from the raw point_2d/label format (tail->head segments + labels), "
             "bypassing normalize_arrows()/convert_to_bbox()",
    )
    parser.add_argument(
        "--normalized", action="store_true",
        help="Coordinates in the JSON are on a 0-1000 scale (see PROMPT_NORMALIZED in inference.py); "
             "rescale them to the target image's actual pixel size before drawing",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    output_path = args.output or f"{Path(args.json_path).stem}_annotated.jpg"
    if args.raw:
        draw_raw(args.json_path, args.image, output_path, normalized=args.normalized)
    else:
        draw(args.json_path, args.image, output_path, normalized=args.normalized)
