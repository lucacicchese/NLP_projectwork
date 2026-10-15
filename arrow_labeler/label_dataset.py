import argparse
import csv
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

RASTER_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
POINT_HALF = 3  # half side of the arrow_start / arrow_end boxes, their center is the keypoint
MAX_WORK_SIDE = 3000  # very large renders are analysed at this size (coordinates are mapped back)
CANDIDATE_CONF = 0.2  # raw candidates are collected below the final threshold, then validated and merged


def rasterize_svgs(images_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for svg in sorted(images_dir.glob("*.svg")):
        png = out_dir / f"{svg.stem}.png"
        if png.exists():
            continue
        cmd = ["inkscape", str(svg), "--export-type=png", f"--export-filename={png}", "--export-background=white", "--export-background-opacity=1"]
        subprocess.run(cmd, capture_output=True)
        print(f"rasterized {svg.name} -> {png.name}" if png.exists() else f"FAILED to rasterize {svg.name}")


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = (img / 257).astype(np.uint8) if img.dtype == np.uint16 else img.astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        return (img[:, :, :3].astype(np.float32) * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    return img


def _detect(model, img, imgsz, conf, offset=(0, 0)):
    res = model.predict(img, imgsz=imgsz, conf=conf, verbose=False, max_det=1000)[0]
    out = []
    if res.keypoints is None:
        return out
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    kpts = res.keypoints.xy.cpu().numpy()
    ox, oy = offset
    for b, c, k, kp in zip(boxes, confs, clss, kpts):
        out.append({
            "box": [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy], "conf": float(c), "cls": int(k),
            "tail": (float(kp[0][0]) + ox, float(kp[0][1]) + oy), "head": (float(kp[1][0]) + ox, float(kp[1][1]) + oy),
        })
    return out


def _tiles(w, h, tile, overlap):
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    ys = list(range(0, max(h - tile, 0) + 1, step))
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    return [(x, y) for y in ys for x in xs]


def _same_arrow(a, b, tol_frac=0.12, tol_min=8.0):
    la = math.dist(a["tail"], a["head"])
    lb = math.dist(b["tail"], b["head"])
    tol = max(tol_min, tol_frac * min(la, lb))
    if math.dist(a["tail"], b["tail"]) <= tol and math.dist(a["head"], b["head"]) <= tol:
        return True
    # opposite direction on the same segment: same arrow only if one of them is a double arrow
    if (a["cls"] == 1 or b["cls"] == 1) and math.dist(a["tail"], b["head"]) <= tol and math.dist(a["head"], b["tail"]) <= tol:
        return True
    # same arrowhead, and the shorter one is a truncated copy lying on the longer one
    if math.dist(a["head"], b["head"]) <= tol_min:
        short, long_ = (a, b) if la <= lb else (b, a)
        if _point_segment_dist(short["tail"], long_["tail"], long_["head"]) <= tol_min and _contained(short["box"], long_["box"]) > 0.8:
            return True
    return False


def _point_segment_dist(p, a, b):
    ax, ay, bx, by = *a, *b
    dx, dy = bx - ax, by - ay
    d2 = dx * dx + dy * dy
    t = 0.0 if d2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / d2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def _contained(inner, outer):
    ix0, iy0, ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1]), min(inner[2], outer[2]), min(inner[3], outer[3])
    area = max(inner[2] - inner[0], 1) * max(inner[3] - inner[1], 1)
    return max(ix1 - ix0, 0) * max(iy1 - iy0, 0) / area


def merge(dets):
    kept = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        if not any(_same_arrow(d, k) for k in kept):
            kept.append(d)
    return kept


class InkCheck:

    def __init__(self, img):
        h, w = img.shape[:2]
        k = int(min(61, max(21, max(h, w) / 50))) | 1
        background = cv2.medianBlur(img, k)  # thin strokes vanish, flat fills and gradients stay
        diff = cv2.absdiff(img, background).max(axis=2)
        self.ink = (diff > 28).astype(np.uint8)
        self.near = cv2.dilate(self.ink, np.ones((5, 5), np.uint8))
        self.joined = cv2.dilate(self.ink, np.ones((9, 9), np.uint8))  # bridges dashes and antialiasing gaps
        self.h, self.w = h, w

    def _window(self, mask, p, r):
        x, y = int(round(p[0])), int(round(p[1]))
        return mask[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1]

    def on_ink(self, p, r=6):
        win = self._window(self.ink, p, r)
        return win.size > 0 and bool(win.any())

    def straight_fraction(self, a, b):
        n = max(int(math.dist(a, b) / 2), 2)
        xs = np.clip(np.linspace(a[0], b[0], n).round().astype(int), 0, self.w - 1)
        ys = np.clip(np.linspace(a[1], b[1], n).round().astype(int), 0, self.h - 1)
        return float(self.near[ys, xs].mean())

    def connected(self, box, tail, head, pad=14, r=6):
        x0, y0 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
        x1, y1 = min(self.w, int(box[2]) + pad), min(self.h, int(box[3]) + pad)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return False
        _, labels = cv2.connectedComponents(self.joined[y0:y1, x0:x1], connectivity=8)

        def ids(p):
            x, y = int(round(p[0])) - x0, int(round(p[1])) - y0
            win = labels[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1]
            return set(np.unique(win).tolist()) - {0}

        return bool(ids(tail) & ids(head))

    def valid(self, d):
        if not (self.on_ink(d["tail"]) and self.on_ink(d["head"])):
            return False
        return self.straight_fraction(d["tail"], d["head"]) >= 0.8 or self.connected(d["box"], d["tail"], d["head"])


def _detect_flipped(model, img, imgsz, conf, offset=(0, 0)):
    w = img.shape[1]
    out = _detect(model, np.ascontiguousarray(img[:, ::-1]), imgsz, conf)
    ox, oy = offset
    for d in out:
        x0, y0, x1, y1 = d["box"]
        d["box"] = [w - x1 + ox, y0 + oy, w - x0 + ox, y1 + oy]
        d["tail"] = (w - d["tail"][0] + ox, d["tail"][1] + oy)
        d["head"] = (w - d["head"][0] + ox, d["head"][1] + oy)
        if d["cls"] == 1 and d["head"][0] < d["tail"][0]:  # double arrows keep the left-most end first
            d["tail"], d["head"] = d["head"], d["tail"]
    return out


def _iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih)


def detect_arrows(model, img, imgsz=1280, conf=0.45, tile_overlap=0.25, tta=True, ink_check=True):
    full_h, full_w = img.shape[:2]
    scale = min(1.0, MAX_WORK_SIDE / max(full_h, full_w))
    if scale < 1.0:
        img = cv2.resize(img, (int(full_w * scale), int(full_h * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    views = [(img, (0, 0), False)]
    if max(h, w) > imgsz * 1.5:
        # large image: arrows get too thin once the whole image is squeezed to imgsz, so also look at tiles
        views += [(img[y : y + imgsz, x : x + imgsz], (x, y), True) for x, y in _tiles(w, h, imgsz, tile_overlap)]

    dets, margin = [], 6
    for view, (x, y), is_tile in views:
        found = _detect(model, view, imgsz, min(conf, CANDIDATE_CONF), offset=(x, y))
        if tta:
            found += _detect_flipped(model, view, imgsz, min(conf, CANDIDATE_CONF), offset=(x, y))
        ch, cw = view.shape[:2]
        for d in found:
            bx0, by0, bx1, by1 = d["box"]
            # drop arrows cut by an inner tile border: the neighbouring tile or the full view has them whole
            cut = is_tile and (
                (bx0 - x < margin and x > 0) or (by0 - y < margin and y > 0)
                or (x + cw - bx1 < margin and x + cw < w) or (y + ch - by1 < margin and y + ch < h)
            )
            if not cut:
                dets.append(d)

    if ink_check:
        check = InkCheck(img)
        valid = [d for d in dets if check.valid(d)]
        # a rejected high-confidence candidate (typically a drifted tail) still vouches for the arrowhead:
        # pass its confidence on to the valid candidate describing the same arrow
        for d in valid:
            d["conf"] = max(o["conf"] for o in dets if math.dist(o["head"], d["head"]) <= 8 and (_iou(o["box"], d["box"]) > 0.3 or _contained(d["box"], o["box"]) > 0.7))
        dets = valid
    dets = [d for d in merge(dets) if d["conf"] >= conf]
    if scale < 1.0:
        for d in dets:
            d["box"] = [v / scale for v in d["box"]]
            d["tail"] = (d["tail"][0] / scale, d["tail"][1] / scale)
            d["head"] = (d["head"][0] / scale, d["head"][1] / scale)
    return dets


def to_rows(dets, w, h):
    rows, arrow_id = [], 0
    for d in sorted(dets, key=lambda d: (d["box"][1], d["box"][0])):
        x0, y0, x1, y1 = (int(round(v)) for v in d["box"])
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        directions = [(d["tail"], d["head"])]
        if d["cls"] == 1:  # double-headed: one arrow per arrowhead
            directions.append((d["head"], d["tail"]))
        for tail, head in directions:
            arrow_id += 1
            rows.append([x0, y0, x1, y1, "arrow", round(d["conf"], 4), arrow_id])
            for (px, py), label in ((tail, "arrow_start"), (head, "arrow_end")):
                px, py = min(max(px, 0), w - 1), min(max(py, 0), h - 1)
                rows.append([round(px - POINT_HALF, 1), round(py - POINT_HALF, 1), round(px + POINT_HALF, 1), round(py + POINT_HALF, 1), label, round(d["conf"], 4), arrow_id])
    return rows


def draw_vis(img, rows, path):
    vis = img.copy()
    scale = max(1, round(max(img.shape[:2]) / 1400))
    by_id = {}
    for r in rows:
        by_id.setdefault(r[6], {})[r[4]] = r
    for aid, g in by_id.items():
        a, st, en = g["arrow"], g["arrow_start"], g["arrow_end"]
        tail = (int((st[0] + st[2]) / 2), int((st[1] + st[3]) / 2))
        head = (int((en[0] + en[2]) / 2), int((en[1] + en[3]) / 2))
        cv2.rectangle(vis, (a[0], a[1]), (a[2], a[3]), (255, 170, 60), scale)
        cv2.line(vis, tail, head, (255, 0, 255), scale, cv2.LINE_AA)
        cv2.circle(vis, tail, 5 * scale, (0, 180, 0), 2 * scale, cv2.LINE_AA)  # green ring = tail
        cv2.circle(vis, head, 4 * scale, (0, 0, 255), -1, cv2.LINE_AA)  # red dot = head
        cv2.putText(vis, f"{a[5]:.2f}", (a[0], max(10, a[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (255, 120, 0), scale, cv2.LINE_AA)
    cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(path))


def main():
    ap = argparse.ArgumentParser(description="Auto-label arrows (bbox + tail + head) with the YOLO-pose model")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True, help="Output folder for the CSV labels")
    ap.add_argument("--vis", default=None, help="Optional output folder for visual checks")
    ap.add_argument("--weights", default=str(Path(__file__).parent / "arrow_pose.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--svg_out", default=None, help="Where to write PNG renders of SVG images (default: <images>/../svg_png)")
    ap.add_argument("--only", nargs="*", default=None, help="Only these image stems")
    a = ap.parse_args()

    images_dir, labels_dir = Path(a.images), Path(a.labels)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if a.vis:
        Path(a.vis).mkdir(parents=True, exist_ok=True)
    svg_dir = Path(a.svg_out) if a.svg_out else images_dir.parent / "svg_png"
    if any(images_dir.glob("*.svg")):
        rasterize_svgs(images_dir, svg_dir)
    model = YOLO(a.weights)

    candidates = list(images_dir.iterdir()) + (list(svg_dir.iterdir()) if svg_dir.exists() else [])
    files = sorted((p for p in candidates if p.suffix.lower() in RASTER_EXT), key=lambda p: (len(p.stem), p.stem))
    n_arrows, skipped = 0, []
    for path in files:
        if a.only and path.stem not in a.only:
            continue
        img = read_image(path)
        if img is None:
            skipped.append(path.name)
            continue
        h, w = img.shape[:2]
        rows = to_rows(detect_arrows(model, img, a.imgsz, a.conf), w, h)
        n_arrows += len(rows) // 3
        with open(labels_dir / f"{path.stem}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["xmin", "ymin", "xmax", "ymax", "label", "conf", "arrow_id"])
            writer.writerows(rows)
        if a.vis:
            draw_vis(img, rows, Path(a.vis) / f"{path.stem}.jpg")
    print(f"labelled {len(files) - len(skipped)} images, {n_arrows} arrows -> {labels_dir}")
    if skipped:
        print("unreadable (not images):", ", ".join(skipped))


if __name__ == "__main__":
    main()
