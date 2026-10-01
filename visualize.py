from pathlib import Path

from PIL import Image, ImageDraw

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


def draw_from_json(image_path, json_path, labels_dir=None, output_path=None, show=False):
    import json
    with open(json_path) as f:
        arrows = json.load(f)
    gt_boxes = None
    if labels_dir:
        gt_boxes = load_gt_boxes(Path(labels_dir) / f"{Path(json_path).stem}.csv")
    return draw_boxes(image_path, arrows, gt_boxes, output_path, show)
