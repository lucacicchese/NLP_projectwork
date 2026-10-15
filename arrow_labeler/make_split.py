import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_dataset import RASTER_EXT, read_image  # noqa: E402


def load_arrows(csv_path):
    groups = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            groups.setdefault(r["arrow_id"], {})[r["label"]] = r
    return [g for g in groups.values() if {"arrow", "arrow_start", "arrow_end"} <= set(g)]


def center(r):
    return (float(r["xmin"]) + float(r["xmax"])) / 2, (float(r["ymin"]) + float(r["ymax"])) / 2


def write_labels(arrows, path, tail_head_rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["xmin", "ymin", "xmax", "ymax", "label", "conf", "arrow_id"])
        for g in arrows:
            a, s, e = g["arrow"], g["arrow_start"], g["arrow_end"]
            if tail_head_rows:
                (tx, ty), (hx, hy) = center(s), center(e)
                w.writerow([round(tx, 1), round(ty, 1), round(hx, 1), round(hy, 1), "arrow", a["conf"], a["arrow_id"]])
            else:
                w.writerow([a["xmin"], a["ymin"], a["xmax"], a["ymax"], "arrow", a["conf"], a["arrow_id"]])
            for r in (s, e):
                w.writerow([r["xmin"], r["ymin"], r["xmax"], r["ymax"], r["label"], r["conf"], r["arrow_id"]])


def export_image(src, dst_dir):
    raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    plain = raw is not None and raw.ndim == 3 and raw.shape[2] == 3 and raw.dtype.name == "uint8"
    if plain and src.suffix.lower() in (".png", ".jpg", ".jpeg"):
        shutil.copyfile(src, dst_dir / f"{src.stem}{src.suffix.lower()}")
    else:
        cv2.imwrite(str(dst_dir / f"{src.stem}.png"), read_image(src))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_unifi")
    ap.add_argument("--labels", default="labels_pose")
    ap.add_argument("--out", default="dataset_unifi_split")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root, out = Path(a.dataset), Path(a.out)
    sources = {}
    for folder in (root / "imgs", root / "svg_png"):
        if folder.exists():
            for p in folder.iterdir():
                if p.suffix.lower() in RASTER_EXT:
                    sources[p.stem] = p
    stems = sorted((p.stem for p in (root / a.labels).glob("*.csv") if p.stem in sources), key=lambda s: (len(s), s))
    random.Random(a.seed).shuffle(stems)
    n_test = round(len(stems) * a.test_frac)
    splits = {"test": sorted(stems[:n_test], key=int), "train": sorted(stems[n_test:], key=int)}
    for split, names in splits.items():
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        n_arrows = 0
        for stem in names:
            arrows = load_arrows(root / a.labels / f"{stem}.csv")
            n_arrows += len(arrows)
            write_labels(arrows, out / split / "labels" / f"{stem}.csv", tail_head_rows=(split == "train"))
            export_image(sources[stem], out / split / "images")
        (out / f"{split}.txt").write_text("\n".join(names) + "\n")
        print(f"{split}: {len(names)} images, {n_arrows} arrows")
