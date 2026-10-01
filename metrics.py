import csv
import math
from pathlib import Path

ARROW_LABELS = {"arrow", "Arrow", "ARROW", "6"}
PCK_TOLERANCE_FRAC = 0.03  # fraction of the image diagonal a predicted point may miss by


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


def load_gt_points(labels_csv):
    labels_csv = Path(labels_csv)
    starts, ends = [], []
    if not labels_csv.exists():
        return starts, ends
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip()
            try:
                cx = (float(row["xmin"]) + float(row["xmax"])) / 2
                cy = (float(row["ymin"]) + float(row["ymax"])) / 2
            except (KeyError, TypeError, ValueError):
                continue
            if label == "arrow_start":
                starts.append((cx, cy))
            elif label == "arrow_end":
                ends.append((cx, cy))

    if not starts and not ends:
        # Fall back to the coarse "arrow" bbox corners when precise start/end
        # markers aren't in this CSV (e.g. older or third-party label sets).
        for x1, y1, x2, y2 in load_gt_boxes(labels_csv):
            starts.append((x1, y1))
            ends.append((x2, y2))
    return starts, ends


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def match_points(gt_points, pred_points, tolerance):
    available = list(gt_points)
    tp, dists = 0, []
    for p in pred_points:
        if not available:
            break
        d, j = min((math.hypot(p[0] - g[0], p[1] - g[1]), j) for j, g in enumerate(available))
        dists.append(d)
        if d <= tolerance:
            tp += 1
            available.pop(j)
    fp = len(pred_points) - tp
    fn = len(available)
    return tp, fp, fn, dists


def evaluate(predictions, labels_dir, image_sizes, tol_frac=PCK_TOLERANCE_FRAC):
    """Point-distance based evaluation (PCK-style): matches predicted tail/head
    points to the nearest unused ground-truth arrow_start/arrow_end point, within
    `tol_frac` of the image diagonal. Box-IoU isn't used here because a model that
    correctly predicts a perfectly straight arrow produces a zero-width bbox, which
    always has IoU 0 regardless of how accurate the prediction is."""
    labels_dir = Path(labels_dir)
    total_tp = total_fp = total_fn = 0
    all_dists, diag_sum = [], 0.0
    for stem, arrows in predictions.items():
        starts, ends = load_gt_points(labels_dir / f"{stem}.csv")
        w, h = image_sizes[stem]
        diag = math.hypot(w, h)
        tolerance = tol_frac * diag

        pred_tails = [tuple(a["tail"]) for a in arrows]
        pred_heads = [tuple(a["head"]) for a in arrows]

        for gt_points, pred_points in ((starts, pred_tails), (ends, pred_heads)):
            tp, fp, fn, dists = match_points(gt_points, pred_points, tolerance)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            all_dists.extend(d / diag for d in dists)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_dist_pct_diag = 100 * sum(all_dists) / len(all_dists) if all_dists else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "mean_dist_pct_diag": mean_dist_pct_diag,
    }
