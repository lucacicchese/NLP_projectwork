import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402
from label_dataset import RASTER_EXT, detect_arrows, read_image  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="training_data/val", help="Folder with images/ and labels/")
    ap.add_argument("--weights", default=str(Path(__file__).parent / "arrow_pose.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--no_ink_check", action="store_true")
    a = ap.parse_args()

    model = YOLO(a.weights)
    predictions, sizes = {}, {}
    for path in sorted((Path(a.dataset) / "images").iterdir()):
        if path.suffix.lower() not in RASTER_EXT:
            continue
        img = read_image(path)
        sizes[path.stem] = (img.shape[1], img.shape[0])
        arrows = []
        for d in detect_arrows(model, img, a.imgsz, a.conf, tta=not a.no_tta, ink_check=not a.no_ink_check):
            arrows.append({"tail": d["tail"], "head": d["head"]})
            if d["cls"] == 1:
                arrows.append({"tail": d["head"], "head": d["tail"]})
        predictions[path.stem] = arrows
    for tol in (0.03, 0.015):
        r = metrics.evaluate(predictions, Path(a.dataset) / "labels", sizes, tol_frac=tol)
        print(f"tol={tol:.3f} diag  P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}  tp={r['tp']} fp={r['fp']} fn={r['fn']}")
