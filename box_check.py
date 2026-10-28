import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MARGIN_FRAC = 0.03
INSIDE_THRESHOLD = 0.75
STRAIGHT_ANGLE_DEG = 15.0
SEGMENT_SAMPLES = 50
MAX_GAP_FRAC = 0.01
FILL_FRAC = 0.01
BACKGROUND_DIFF = 16

ARROW_LABELS = {"arrow", "Arrow", "ARROW", "6"}


@dataclass
class Box:
    center: tuple
    axis: tuple
    half_len: float
    half_wid: float
    kind: str

    @property
    def normal(self):
        return -self.axis[1], self.axis[0]

    def corners(self):
        (cx, cy), (ux, uy), (nx, ny) = self.center, self.axis, self.normal
        return [
            (cx + sa * self.half_len * ux + sb * self.half_wid * nx, cy + sa * self.half_len * uy + sb * self.half_wid * ny)
            for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]

    def contains(self, points):
        d = np.asarray(points, dtype=float) - np.asarray(self.center)
        along = d @ np.asarray(self.axis)
        across = d @ np.asarray(self.normal)
        return (np.abs(along) <= self.half_len + 1e-6) & (np.abs(across) <= self.half_wid + 1e-6)


def orientation(tail, head, straight_deg=STRAIGHT_ANGLE_DEG):
    dx, dy = abs(head[0] - tail[0]), abs(head[1] - tail[1])
    angle = math.degrees(math.atan2(dy, dx))
    if angle <= straight_deg:
        return "horizontal"
    if angle >= 90 - straight_deg:
        return "vertical"
    return "diagonal"


def arrow_box(tail, head, img_w, img_h, margin_frac=MARGIN_FRAC, straight_deg=STRAIGHT_ANGLE_DEG):
    mx, my = margin_frac * img_w, margin_frac * img_h
    kind = orientation(tail, head, straight_deg)
    x1, x2 = sorted((tail[0], head[0]))
    y1, y2 = sorted((tail[1], head[1]))
    center = ((x1 + x2) / 2, (y1 + y2) / 2)

    if kind == "horizontal":
        return Box(center, (1.0, 0.0), (x2 - x1) / 2 + mx, (y2 - y1) / 2 + my, kind)
    if kind == "vertical":
        return Box(center, (0.0, 1.0), (y2 - y1) / 2 + my, (x2 - x1) / 2 + mx, kind)
    return Box(center, (1.0, 0.0), (x2 - x1) / 2 + mx, (y2 - y1) / 2 + my, kind)


def fraction_inside(tail, head, box, n=SEGMENT_SAMPLES):
    t = np.linspace(0.0, 1.0, n)[:, None]
    points = np.asarray(tail, dtype=float) * (1 - t) + np.asarray(head, dtype=float) * t
    return float(box.contains(points).mean())


def margin_for_epoch(epoch, epochs, start=MARGIN_FRAC, end=MARGIN_FRAC):
    if epochs <= 1:
        return end
    return start + (end - start) * min(epoch, epochs - 1) / (epochs - 1)


def center(row):
    return (float(row["xmin"]) + float(row["xmax"])) / 2, (float(row["ymin"]) + float(row["ymax"])) / 2


def load_gt_arrows(labels_csv):
    labels_csv = Path(labels_csv)
    if not labels_csv.exists():
        return []
    with open(labels_csv) as f:
        rows = list(csv.DictReader(f))

    def label(r):
        return (r.get("label") or "").strip()

    starts = [r for r in rows if label(r) == "arrow_start"]
    ends = [r for r in rows if label(r) == "arrow_end"]

    def inside(box_row, marker, tol=5.0):
        x, y = center(marker)
        return (float(box_row["xmin"]) - tol <= x <= float(box_row["xmax"]) + tol
                and float(box_row["ymin"]) - tol <= y <= float(box_row["ymax"]) + tol)

    arrows = []
    for row in rows:
        if label(row) and label(row) not in ARROW_LABELS:
            continue
        try:
            x1, y1, x2, y2 = (float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"]))
        except (KeyError, TypeError, ValueError):
            continue
        arrow_id = (row.get("arrow_id") or "").strip()
        if arrow_id:
            s = [r for r in starts if (r.get("arrow_id") or "").strip() == arrow_id]
            e = [r for r in ends if (r.get("arrow_id") or "").strip() == arrow_id]
        else:
            s = [r for r in starts if inside(row, r)]
            e = [r for r in ends if inside(row, r)]
        if len(s) == 1 and len(e) == 1:
            arrows.append((center(s[0]), center(e[0])))
        elif s and e:
            corners = [((x1, y1), (x2, y2)), ((x2, y2), (x1, y1)), ((x1, y2), (x2, y1)), ((x2, y1), (x1, y2))]
            _, tail, head = min(
                (math.dist(center(a), ca) + math.dist(center(b), cb), center(a), center(b))
                for a in s for b in e for ca, cb in corners
            )
            arrows.append((tail, head))
        else:
            arrows.append(((x1, y1), (x2, y2)))
    return arrows


def box_match(gt_arrows, pred_arrows, img_w, img_h, margin_frac=MARGIN_FRAC, threshold=INSIDE_THRESHOLD):
    boxes = [arrow_box(t, h, img_w, img_h, margin_frac) for t, h in gt_arrows]
    pairs = []
    for j, p in enumerate(pred_arrows):
        for i, box in enumerate(boxes):
            pairs.append((fraction_inside(p["tail"], p["head"], box), i, j))
    pairs.sort(reverse=True)

    used_gt, used_pred, matches = set(), set(), []
    for frac, i, j in pairs:
        if frac < threshold:
            break
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        (gt_t, gt_h), p = gt_arrows[i], pred_arrows[j]
        same = math.dist(p["tail"], gt_t) + math.dist(p["head"], gt_h)
        swapped = math.dist(p["tail"], gt_h) + math.dist(p["head"], gt_t)
        matches.append({"gt": i, "pred": j, "inside": frac, "direction_ok": same <= swapped})

    best = [max((f for f, _, jj in pairs if jj == j), default=0.0) for j in range(len(pred_arrows))]
    return {
        "tp": len(matches), "fp": len(pred_arrows) - len(matches), "fn": len(gt_arrows) - len(matches),
        "matches": matches, "best_inside": best, "boxes": boxes,
    }


def evaluate_box(predictions, labels_dir, image_sizes, margin_frac=MARGIN_FRAC, threshold=INSIDE_THRESHOLD):
    labels_dir = Path(labels_dir)
    tp = fp = fn = direction_ok = 0
    for stem, arrows in predictions.items():
        w, h = image_sizes[stem]
        r = box_match(load_gt_arrows(labels_dir / f"{stem}.csv"), arrows, w, h, margin_frac, threshold)
        tp, fp, fn = tp + r["tp"], fp + r["fp"], fn + r["fn"]
        direction_ok += sum(m["direction_ok"] for m in r["matches"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
        "direction_ok": direction_ok, "margin_frac": margin_frac, "threshold": threshold,
    }


def foreground_mask(image, min_diff=BACKGROUND_DIFF):
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    quantized = (rgb // 8).reshape(-1, 3)
    codes = quantized[:, 0] * 1024 + quantized[:, 1] * 32 + quantized[:, 2]
    mode = np.bincount(codes).argmax()
    background = rgb.reshape(-1, 3)[codes == mode].mean(axis=0)
    return np.abs(rgb - background).max(axis=2) > min_diff


def stroke_mask(mask, fill_frac=FILL_FRAC):
    from scipy import ndimage

    h, w = mask.shape
    k = max(9, round(fill_frac * max(w, h))) | 1
    return mask & ~ndimage.binary_opening(mask, structure=np.ones((k, k), dtype=bool))


def any_hit_per_row(mask, pts, valid=True):
    h, w = mask.shape
    xs, ys = np.rint(pts[..., 0]).astype(int), np.rint(pts[..., 1]).astype(int)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & valid
    hits = np.zeros(xs.shape, dtype=bool)
    hits[valid] = mask[ys[valid], xs[valid]]
    return hits.any(axis=1)


def scan_hits(mask, box, start, end):
    c, u, n = np.asarray(box.center), np.asarray(box.axis), np.asarray(box.normal)
    a0, a1 = sorted(((np.asarray(start) - c) @ u, (np.asarray(end) - c) @ u))
    along = np.linspace(a0, a1, max(int(a1 - a0) + 1, 2))
    across = np.arange(-box.half_wid, box.half_wid + 1.0, 1.0)
    pts = c + along[:, None, None] * u + across[None, :, None] * n
    return any_hit_per_row(mask, pts)


def scan_band(mask, box, start, end, half_wid):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    length = float(np.hypot(*(end - start)))
    u = (end - start) / length
    n = np.array([-u[1], u[0]])
    along = np.linspace(0.0, length, max(int(length) + 1, 2))
    across = np.arange(-half_wid, half_wid + 1.0, 1.0)
    pts = start + along[:, None, None] * u + across[None, :, None] * n
    return any_hit_per_row(mask, pts, box.contains(pts))


def longest_run(hits, max_gap):
    best = run = gap = 0
    for hit in hits:
        if hit:
            run += gap + 1
            gap = 0
        elif run and gap < max_gap:
            gap += 1
        else:
            run = gap = 0
        best = max(best, run)
    return best / len(hits) if len(hits) else 0.0


def path_scores(mask, tail, head, margin_frac=MARGIN_FRAC, max_gap_frac=MAX_GAP_FRAC):
    h, w = mask.shape
    max_gap = max(1, round(max_gap_frac * math.hypot(w, h)))
    box = arrow_box(tail, head, w, h, margin_frac)
    tail, head = tuple(tail), tuple(head)

    paths = {"straight": [tail, head]}
    if box.kind == "diagonal":
        paths["elbow_hv"] = [tail, (head[0], tail[1]), head]
        paths["elbow_vh"] = [tail, (tail[0], head[1]), head]

    def scan(a, b):
        leg_box = arrow_box(a, b, w, h, margin_frac)
        if leg_box.kind != "diagonal":
            hits = scan_hits(mask, leg_box, a, b)
            going_back = (b[0] - a[0]) * leg_box.axis[0] + (b[1] - a[1]) * leg_box.axis[1] < 0
            return hits[::-1] if going_back else hits
        ux, uy = (b[0] - a[0]) / math.dist(a, b), (b[1] - a[1]) / math.dist(a, b)
        return scan_band(mask, leg_box, a, b, abs(uy) * margin_frac * w + abs(ux) * margin_frac * h)

    results = []
    for name, points in paths.items():
        hits = np.concatenate([scan(a, b) for a, b in zip(points, points[1:])])
        results.append({"path": name, "points": points, "coverage": float(hits.mean()), "continuity": longest_run(hits, max_gap)})
    return box.kind, results


def pixel_support(mask, tail, head, margin_frac=MARGIN_FRAC, max_gap_frac=MAX_GAP_FRAC):
    kind, results = path_scores(mask, tail, head, margin_frac, max_gap_frac)
    best = max(results, key=lambda r: r["continuity"])
    return {"kind": kind, **{k: v for k, v in best.items() if k != "points"}}


def choose_path(mask, tail, head, margin_frac=MARGIN_FRAC, threshold=INSIDE_THRESHOLD):
    _, results = path_scores(mask, tail, head, margin_frac)
    elbows = [r for r in results if r["path"] != "straight"]
    best = max(elbows, key=lambda r: r["continuity"], default=None)
    if best is None or best["continuity"] < threshold:
        best = results[0]
    return best["path"], best["points"]


def filter_arrows(image, arrows, margin_frac=MARGIN_FRAC, threshold=INSIDE_THRESHOLD):
    mask = foreground_mask(image)
    kept, report = [], []
    for arrow in arrows:
        s = pixel_support(mask, arrow["tail"], arrow["head"], margin_frac)
        s["kept"] = s["continuity"] >= threshold
        report.append(s)
        if s["kept"]:
            kept.append(arrow)
    return kept, report


def find_image(images_dir, stem):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        p = Path(images_dir) / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def draw_check(image, gt_arrows, arrows, match, support, output_path):
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    lw = max(2, round(0.003 * max(w, h)))
    for box in match["boxes"]:
        draw.polygon(box.corners(), outline="#2E7D32", width=lw)
    accepted = {m["pred"] for m in match["matches"]}
    for j, (arrow, s) in enumerate(zip(arrows, support)):
        pbox = arrow_box(arrow["tail"], arrow["head"], w, h)
        draw.polygon(pbox.corners(), outline="#FB8C00" if s["kept"] else "#9E9E9E", width=max(1, lw // 2))
        color = "#1565C0" if j in accepted else "#C62828"
        draw.line([tuple(arrow["tail"]), tuple(arrow["head"])], fill=color, width=lw)
        r = 2 * lw
        hx, hy = arrow["head"]
        draw.ellipse([hx - r, hy - r, hx + r, hy + r], fill=color)
    img.save(output_path)


def run_offline(pred_json_dir, dataset_dir, output_dir=None, margin_frac=MARGIN_FRAC, threshold=INSIDE_THRESHOLD):
    import metrics

    pred_json_dir, dataset_dir = Path(pred_json_dir), Path(dataset_dir)
    images_dir, labels_dir = dataset_dir / "images", dataset_dir / "labels"
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    predictions, filtered, sizes, per_image = {}, {}, {}, {}
    for json_path in sorted(pred_json_dir.glob("*.json")):
        image_path = find_image(images_dir, json_path.stem)
        if image_path is None:
            continue
        arrows = json.loads(json_path.read_text())
        image = Image.open(image_path)
        w, h = image.size
        kept, support = filter_arrows(image, arrows, margin_frac, threshold)
        gt = load_gt_arrows(labels_dir / f"{json_path.stem}.csv")
        match = box_match(gt, arrows, w, h, margin_frac, threshold)
        predictions[json_path.stem], filtered[json_path.stem], sizes[json_path.stem] = arrows, kept, (w, h)
        per_image[json_path.stem] = {"gt": len(gt), "pred": len(arrows), "box_tp": match["tp"], "kept": len(kept)}
        if output_dir:
            draw_check(image, gt, arrows, match, support, Path(output_dir) / f"{json_path.stem}.png")

    return {
        "pck": metrics.evaluate(predictions, labels_dir, sizes),
        "pck_after_pixel_filter": metrics.evaluate(filtered, labels_dir, sizes),
        "box": evaluate_box(predictions, labels_dir, sizes, margin_frac, threshold),
        "box_after_pixel_filter": evaluate_box(filtered, labels_dir, sizes, margin_frac, threshold),
        "per_image": per_image,
    }


def main():
    parser = argparse.ArgumentParser(description="Box checks on saved predictions (no model needed)")
    parser.add_argument("--pred", required=True, help="Folder with the per-image prediction JSON files")
    parser.add_argument("--dataset", required=True, help="Dataset split dir with images/ and labels/ subdirs")
    parser.add_argument("--output_dir", default=None, help="Where to save the annotated images (optional)")
    parser.add_argument("--margin_frac", type=float, default=MARGIN_FRAC)
    parser.add_argument("--threshold", type=float, default=INSIDE_THRESHOLD)
    args = parser.parse_args()

    result = run_offline(args.pred, args.dataset, args.output_dir, args.margin_frac, args.threshold)
    print(f"{'':28s} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    for key in ("pck", "pck_after_pixel_filter", "box", "box_after_pixel_filter"):
        r = result[key]
        print(f"{key:28s} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5}")
    print(f"box matches with the right direction: {result['box']['direction_ok']}/{result['box']['tp']}")
    if args.output_dir:
        with open(Path(args.output_dir) / "box_check.json", "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
