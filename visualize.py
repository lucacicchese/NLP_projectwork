import math
from pathlib import Path

from PIL import Image, ImageDraw

import box_check
from inference import convert_to_bbox
from metrics import load_gt_boxes

PRED_COLOR = "#FF1493"
GT_COLOR = "#2196F3"


def draw_boxes(image_path, arrows=None, gt_boxes=None, output_path=None, show=False):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for x1, y1, x2, y2 in gt_boxes or []:
        draw.rectangle([(x1, y1), (x2, y2)], outline=GT_COLOR, width=3)

    for arrow in arrows or []:
        x1, y1, x2, y2 = convert_to_bbox(arrow["tail"], arrow["head"], w, h)
        draw.rectangle([(x1, y1), (x2, y2)], outline=PRED_COLOR, width=3)
        draw.line([tuple(arrow["tail"]), tuple(arrow["head"])], fill=PRED_COLOR, width=3)

    if gt_boxes:
        draw.text((5, 5), "ground truth", fill=GT_COLOR)
        draw.text((5, 19), "prediction", fill=PRED_COLOR)

    if output_path:
        img.save(output_path)
    if show:
        img.show()
    return img


def arrowhead(draw, tip, prev, size, color):
    length = math.dist(prev, tip)
    if length == 0:
        return
    ux, uy = (tip[0] - prev[0]) / length, (tip[1] - prev[1]) / length
    size = min(size, 0.6 * length)
    bx, by = tip[0] - size * ux, tip[1] - size * uy
    half = 0.5 * size
    draw.polygon([tip, (bx - half * uy, by + half * ux), (bx + half * uy, by - half * ux)], fill=color)


def draw_arrows(image_path, arrows=None, gt_boxes=None, output_path=None, show=False, route=True):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    lw = max(2, round(0.003 * max(w, h)))
    head_size = max(10, 0.02 * max(w, h))
    mask = box_check.stroke_mask(box_check.foreground_mask(img)) if route else None

    for x1, y1, x2, y2 in gt_boxes or []:
        draw.rectangle([(x1, y1), (x2, y2)], outline=GT_COLOR, width=lw)

    for i, arrow in enumerate(arrows or [], start=1):
        tail, head = tuple(arrow["tail"]), tuple(arrow["head"])
        points = box_check.choose_path(mask, tail, head)[1] if route else [tail, head]
        draw.line(points, fill=PRED_COLOR, width=lw, joint="curve")
        arrowhead(draw, head, points[-2], head_size, PRED_COLOR)
        r = 2 * lw
        draw.ellipse([tail[0] - r, tail[1] - r, tail[0] + r, tail[1] + r], fill="white", outline=PRED_COLOR, width=lw)
        draw.text((tail[0] + r + 2, tail[1] + r), str(i), fill=PRED_COLOR)

    legend = ([("ground truth", GT_COLOR)] if gt_boxes else []) + [("prediction: o tail, > head", PRED_COLOR)]
    draw.rectangle([0, 0, 170, 6 + 14 * len(legend)], fill="white")
    for k, (text, color) in enumerate(legend):
        draw.text((5, 4 + 14 * k), text, fill=color)

    if output_path:
        img.save(output_path)
    if show:
        img.show()
    return img


def draw_from_json(image_path, json_path, labels_dir=None, output_path=None, show=False):
    import json
    with open(json_path) as f:
        arrows = json.load(f)
    gt_boxes = None
    if labels_dir:
        gt_boxes = load_gt_boxes(Path(labels_dir) / f"{Path(json_path).stem}.csv")
    return draw_boxes(image_path, arrows, gt_boxes, output_path, show)
