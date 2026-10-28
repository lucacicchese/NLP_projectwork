import argparse
import csv
import math
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import box_check as bc  # noqa: E402

RASTER_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
POINT_HALF = 3
JOIN_TOL_FRAC = 0.03
COLLINEAR_COS = 0.9
DUP_MIN_COS = 0.94
REVERSED_MIN_RUN = 3


def load_rgb(path):
    try:
        img = Image.open(path)
    except Image.UnidentifiedImageError:
        from label_dataset import read_image
        return Image.fromarray(read_image(path)[:, :, ::-1].copy())
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, img).convert("RGB")
    return img.convert("RGB")


def load_labels(csv_path):
    groups = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            groups.setdefault(r["arrow_id"], {})[r["label"]] = r
    arrows = []
    for aid, g in groups.items():
        if not {"arrow", "arrow_start", "arrow_end"} <= set(g):
            continue
        a = g["arrow"]
        arrows.append({
            "ids": [aid], "box": [float(a["xmin"]), float(a["ymin"]), float(a["xmax"]), float(a["ymax"])],
            "conf": float(a.get("conf") or 1.0), "tail": bc.center(g["arrow_start"]), "head": bc.center(g["arrow_end"]),
            "double": False,
        })
    merged, used = [], set()
    for i, a in enumerate(arrows):
        if i in used:
            continue
        for j in range(i + 1, len(arrows)):
            b = arrows[j]
            if j not in used and a["box"] == b["box"] and close(a["tail"], b["head"], 1.5) and close(a["head"], b["tail"], 1.5):
                a = {**a, "ids": a["ids"] + b["ids"], "double": True}
                used.add(j)
                break
        merged.append(a)
    return merged


def write_labels(arrows, path, img_w, img_h):
    rows, arrow_id = [], 0
    for a in sorted(arrows, key=lambda a: (a["box"][1], a["box"][0])):
        x0, y0, x1, y1 = (int(round(v)) for v in a["box"])
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img_w, x1), min(img_h, y1)
        directions = [(a["tail"], a["head"])] + ([(a["head"], a["tail"])] if a["double"] else [])
        for tail, head in directions:
            arrow_id += 1
            rows.append([x0, y0, x1, y1, "arrow", round(a["conf"], 4), arrow_id])
            for (px, py), label in ((tail, "arrow_start"), (head, "arrow_end")):
                px, py = min(max(px, 0), img_w - 1), min(max(py, 0), img_h - 1)
                rows.append([round(px - POINT_HALF, 1), round(py - POINT_HALF, 1), round(px + POINT_HALF, 1), round(py + POINT_HALF, 1), label, round(a["conf"], 4), arrow_id])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["xmin", "ymin", "xmax", "ymax", "label", "conf", "arrow_id"])
        w.writerows(rows)


def close(p, q, tol):
    return math.dist(p, q) <= tol


def direction(a):
    dx, dy = a["head"][0] - a["tail"][0], a["head"][1] - a["tail"][1]
    length = math.hypot(dx, dy)
    return (dx / length, dy / length) if length else (0.0, 0.0)


def union_box(a, b):
    return [min(a["box"][0], b["box"][0]), min(a["box"][1], b["box"][1]), max(a["box"][2], b["box"][2]), max(a["box"][3], b["box"][3])]


def connected_in_box(mask, a, pad=6, r=6):
    h, w = mask.shape
    x0, y0 = max(0, int(a["box"][0]) - pad), max(0, int(a["box"][1]) - pad)
    x1, y1 = min(w, int(a["box"][2]) + pad + 1), min(h, int(a["box"][3]) + pad + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    labels, _ = ndimage.label(ndimage.binary_dilation(mask[y0:y1, x0:x1], iterations=2), structure=np.ones((3, 3)))

    def ids(p):
        x, y = int(round(p[0])) - x0, int(round(p[1])) - y0
        win = labels[max(0, y - r): y + r + 1, max(0, x - r): x + r + 1]
        return set(np.unique(win).tolist()) - {0}

    return bool(ids(a["tail"]) & ids(a["head"]))


def check_no_stroke(mask, arrows, margin_frac, threshold):
    kept, issues = [], []
    for a in arrows:
        s = bc.pixel_support(mask, a["tail"], a["head"], margin_frac)
        if s["continuity"] >= threshold or connected_in_box(mask, a):
            kept.append(a)
        else:
            issues.append((a, "no_stroke", "removed", f"continuity={s['continuity']:.2f}"))
    return kept, issues


def check_split(arrows, tol, merge=True):
    kept, issues, used = [], [], set()
    for i, a in enumerate(arrows):
        if i in used or a["double"]:
            continue
        for j in range(i + 1, len(arrows)):
            b = arrows[j]
            if j in used or b["double"] or not close(a["tail"], b["tail"], tol):
                continue
            da, db = direction(a), direction(b)
            if da[0] * db[0] + da[1] * db[1] > -COLLINEAR_COS:
                continue
            merged = {
                "ids": a["ids"] + b["ids"], "box": union_box(a, b), "conf": max(a["conf"], b["conf"]),
                "tail": b["head"], "head": a["head"], "double": True,
            }
            used.update((i, j))
            kept += [merged] if merge else [a, b]
            action = "merged" if merge else "reported"
            issues.append((a, "split", action, f"with arrow {'/'.join(b['ids'])}"))
            issues.append((b, "split", action, f"with arrow {'/'.join(a['ids'])}"))
            break
    kept += [a for i, a in enumerate(arrows) if i not in used]
    return kept, issues


def check_duplicates(arrows, img_w, img_h, margin_frac, threshold, tol, min_cos=DUP_MIN_COS):
    kept, issues = [], []
    for a in sorted(arrows, key=lambda a: -a["conf"]):
        dup_of = None
        for k in kept:
            box = bc.arrow_box(k["tail"], k["head"], img_w, img_h, margin_frac)
            if bc.fraction_inside(a["tail"], a["head"], box) < threshold:
                continue
            same_dir = (close(a["head"], k["head"], tol) or close(a["tail"], k["tail"], tol))
            either_way = (k["double"] or a["double"]) and (close(a["head"], k["tail"], tol) or close(a["tail"], k["head"], tol))
            da, dk = direction(a), direction(k)
            parallel = abs(da[0] * dk[0] + da[1] * dk[1]) >= min_cos
            if (same_dir or either_way) and parallel:
                dup_of = k
                break
        if dup_of is None:
            kept.append(a)
        else:
            issues.append((a, "duplicate", "removed", f"of arrow {'/'.join(dup_of['ids'])} (conf {dup_of['conf']:.2f})"))
    return kept, issues


def cross_widths(mask, a, b, d0, d1, half=40):
    length = math.dist(a, b)
    u = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
    n = (-u[1], u[0])
    h, w = mask.shape
    offs = np.arange(-half, half + 1)
    out = []
    for d in np.arange(d0, d1, 1.0):
        cx, cy = a[0] + d * u[0], a[1] + d * u[1]
        xs = np.rint(cx + offs * n[0]).astype(int)
        ys = np.rint(cy + offs * n[1]).astype(int)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        line = np.zeros(len(offs), dtype=bool)
        line[ok] = mask[ys[ok], xs[ok]]
        runs, _ = ndimage.label(line)
        near_axis = runs[half - 2: half + 3]
        near_axis = near_axis[near_axis > 0]
        out.append(int((runs == near_axis[0]).sum()) if len(near_axis) else 0)
    return np.array(out)


def arrowhead_run(mask, end, other, line_w):
    length = math.dist(end, other)
    window = min(max(12.0, 0.3 * length), 30.0)
    wide = cross_widths(mask, end, other, 1.0, window) >= 2 * line_w + 2
    return max((len(list(group)) for flag, group in groupby(wide) if flag), default=0)


def check_reversed(mask, arrows, fix):
    kept, issues = [], []
    for a in arrows:
        length = math.dist(a["tail"], a["head"])
        if a["double"] or length < 25:
            kept.append(a)
            continue
        middle = cross_widths(mask, a["tail"], a["head"], 0.4 * length, 0.6 * length)
        middle = middle[middle > 0]
        line_w = float(np.median(middle)) if len(middle) else 2.0
        head_run = arrowhead_run(mask, a["head"], a["tail"], line_w)
        tail_run = arrowhead_run(mask, a["tail"], a["head"], line_w)
        if tail_run >= REVERSED_MIN_RUN and head_run < REVERSED_MIN_RUN:
            detail = f"arrowhead-like run at tail={tail_run}px, at head={head_run}px"
            if fix:
                a = {**a, "tail": a["head"], "head": a["tail"]}
            issues.append((a, "reversed", "swapped" if fix else "reported", detail))
        kept.append(a)
    return kept, issues


def clean_image(image, arrows, margin_frac=bc.MARGIN_FRAC, threshold=bc.INSIDE_THRESHOLD, merge_split=True, fix_reversed=False, checks=("no_stroke", "split", "duplicate", "reversed"), dup_min_cos=DUP_MIN_COS):
    w, h = image.size
    mask = bc.foreground_mask(image)
    tol = JOIN_TOL_FRAC * math.hypot(w, h)
    issues = []
    if "no_stroke" in checks:
        arrows, found = check_no_stroke(mask, arrows, margin_frac, threshold)
        issues += found
    if "split" in checks:
        arrows, found = check_split(arrows, tol, merge_split)
        issues += found
    if "duplicate" in checks:
        arrows, found = check_duplicates(arrows, w, h, margin_frac, threshold, tol, dup_min_cos)
        issues += found
    if "reversed" in checks:
        arrows, found = check_reversed(mask, arrows, fix_reversed)
        issues += found
    return arrows, issues


COLORS = {"kept": "#2E7D32", "removed": "#C62828", "merged": "#7B1FA2", "reported": "#EF6C00", "swapped": "#EF6C00"}


def draw(image, original, cleaned, issues, path):
    img = image.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    lw = max(2, round(0.003 * max(w, h)))
    r = 3 * lw

    def arrow(a, color, width):
        d.line([tuple(a["tail"]), tuple(a["head"])], fill=color, width=width)
        tx, ty = a["tail"]
        hx, hy = a["head"]
        d.ellipse([tx - r, ty - r, tx + r, ty + r], outline=color, width=lw)
        d.ellipse([hx - r, hy - r, hx + r, hy + r], fill=color)
        if a["double"]:
            d.ellipse([tx - r / 2, ty - r / 2, tx + r / 2, ty + r / 2], fill=color)

    for a in cleaned:
        arrow(a, COLORS["kept"], lw)
    for a, issue, action, _ in issues:
        arrow(a, COLORS[action], 2 * lw)
        d.text((a["head"][0] + r + 2, a["head"][1] - r), f"{issue} {'/'.join(a['ids'])}", fill=COLORS[action])
    img.save(path)


def find_images(root):
    sources = {}
    for folder in (root / "imgs", root / "svg_png"):
        if folder.exists():
            for p in folder.iterdir():
                if p.suffix.lower() in RASTER_EXT:
                    sources[p.stem] = p
    return sources


def main():
    parser = argparse.ArgumentParser(description="Clean labels_pose CSVs written by label_dataset.py")
    parser.add_argument("--dataset", default="dataset_unifi", help="Dataset root with imgs/ (and svg_png/) and the labels folder")
    parser.add_argument("--labels", default="labels_pose", help="Labels folder inside --dataset")
    parser.add_argument("--out", default="labels_pose_clean", help="Output folder inside --dataset (never the input one)")
    parser.add_argument("--vis", action="store_true", help="Draw the images with at least one issue in <out>_vis/")
    parser.add_argument("--checks", nargs="+", default=["no_stroke", "split", "duplicate", "reversed"])
    parser.add_argument("--no_merge", action="store_true", help="Only report split double-headed arrows, don't merge them")
    parser.add_argument("--fix_reversed", action="store_true", help="Swap tail and head of the arrows flagged as reversed")
    parser.add_argument("--margin_frac", type=float, default=bc.MARGIN_FRAC)
    parser.add_argument("--threshold", type=float, default=bc.INSIDE_THRESHOLD)
    parser.add_argument("--dup_min_cos", type=float, default=DUP_MIN_COS, help="0 = ignore the angle between duplicates")
    parser.add_argument("--only", nargs="*", default=None, help="Only these image stems")
    args = parser.parse_args()

    root = Path(args.dataset)
    labels_dir, out_dir = root / args.labels, root / args.out
    if out_dir.resolve() == labels_dir.resolve():
        sys.exit("--out must be different from --labels")
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = root / f"{args.out}_vis"
    if args.vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    sources = find_images(root)
    report, counts, n_before, n_after, missing = [], Counter(), 0, 0, []
    for csv_path in sorted(labels_dir.glob("*.csv"), key=lambda p: (len(p.stem), p.stem)):
        stem = csv_path.stem
        if args.only and stem not in args.only:
            continue
        if stem not in sources:
            missing.append(stem)
            continue
        image = load_rgb(sources[stem])
        arrows = load_labels(csv_path)
        if not arrows:
            write_labels([], out_dir / csv_path.name, *image.size)
            continue
        cleaned, issues = clean_image(
            image, arrows, args.margin_frac, args.threshold, merge_split=not args.no_merge, fix_reversed=args.fix_reversed, checks=args.checks, dup_min_cos=args.dup_min_cos,
        )
        write_labels(cleaned, out_dir / csv_path.name, *image.size)
        n_before += sum(1 + x["double"] for x in arrows)
        n_after += sum(1 + x["double"] for x in cleaned)
        for arrow, issue, action, detail in issues:
            report.append([stem, "/".join(arrow["ids"]), issue, action, f"{arrow['conf']:.2f}", detail])
            counts[(issue, action)] += 1
        if args.vis and issues:
            draw(image, arrows, cleaned, issues, vis_dir / f"{stem}.jpg")

    with open(out_dir / "report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "arrow_id", "issue", "action", "conf", "detail"])
        w.writerows(report)
    print(f"arrows (one per direction): {n_before} -> {n_after}")
    for (issue, action), n in sorted(counts.items()):
        print(f"  {issue:10s} {action:9s} {n}")
    print(f"{len({r[0] for r in report})} images with issues; details in {out_dir / 'report.csv'}")
    if missing:
        print(f"no image found for {len(missing)} label files: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")


if __name__ == "__main__":
    main()
