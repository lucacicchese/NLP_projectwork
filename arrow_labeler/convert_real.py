import argparse
import csv
import os
from pathlib import Path

from PIL import Image


def _box(r):
    return float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])


def _center(r):
    x0, y0, x1, y1 = _box(r)
    return (x0 + x1) / 2, (y0 + y1) / 2


def _axis_score(lo, hi, a, b, pad):
    if hi - lo < 16:
        return 1.0
    s0, s1 = min(a, b) - pad, max(a, b) + pad
    inter = min(hi + pad, s1) - max(lo - pad, s0)
    union = max(hi + pad, s1) - min(lo - pad, s0)
    return max(inter, 0.0) / union


def pair_arrows(rows, tol=6.0, pad=4.0):
    arrows = [_box(r) for r in rows if r["label"] == "arrow"]
    starts = [_center(r) for r in rows if r["label"] == "arrow_start"]
    ends = [_center(r) for r in rows if r["label"] == "arrow_end"]
    out = []
    for box in arrows:
        inside = lambda p: box[0] - tol <= p[0] <= box[2] + tol and box[1] - tol <= p[1] <= box[3] + tol
        best, best_score = None, 0.0
        for st in filter(inside, starts):
            for en in filter(inside, ends):
                score = _axis_score(box[0], box[2], st[0], en[0], pad) * _axis_score(box[1], box[3], st[1], en[1], pad)
                if score > best_score:
                    best, best_score = (st, en), score
        out.append((box, best if best_score > 0.6 else None))
    return out


def flowlearn_arrows(rows):
    out = []
    for i in range(0, len(rows) - 2, 3):
        s, e, a = rows[i : i + 3]
        if (s["label"], e["label"], a["label"]) == ("arrow_start", "arrow_end", "arrow"):
            out.append((_box(a), (_center(s), _center(e))))
    return out


def convert(src, out, split, prefix=None, max_images=None):
    src, out = Path(src), Path(out)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_img = n_arrow = n_nokp = 0
    for csv_path in sorted((src / "labels").glob("*.csv")):
        if prefix and not csv_path.stem.startswith(prefix):
            continue
        img_path = src / "images" / f"{csv_path.stem}.jpg"
        if not img_path.exists():
            continue
        if max_images and n_img >= max_images:
            break
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        arrows = flowlearn_arrows(rows) if csv_path.stem.startswith("flowlearn_") else pair_arrows(rows)
        W, H = Image.open(img_path).size
        lines = []
        for (x0, y0, x1, y1), kp in arrows:
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            bw, bh = max(x1 - x0, 2), max(y1 - y0, 2)
            line = f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {bw / W:.6f} {bh / H:.6f} "
            if kp:
                (tx, ty), (hx, hy) = kp
                line += f"{tx / W:.6f} {ty / H:.6f} 2 {hx / W:.6f} {hy / H:.6f} 2"
            else:
                line += "0 0 0 0 0 0"
                n_nokp += 1
            lines.append(line)
        n_arrow += len(lines)
        n_img += 1
        dst = out / "images" / split / img_path.name
        if not dst.exists():
            os.symlink(img_path.resolve(), dst)
        (out / "labels" / split / f"{csv_path.stem}.txt").write_text("\n".join(lines))
    print(f"{src} [{prefix}] -> {split}: {n_img} images, {n_arrow} arrows ({n_nokp} without keypoints)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--training_data", default="training_data")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    convert(Path(a.training_data) / "train", a.out, "train", prefix="flowchart")
    convert(Path(a.training_data) / "train", a.out, "train", prefix="flowlearn")
    convert(Path(a.training_data) / "val", a.out, "val_fc", prefix="flowchart")
