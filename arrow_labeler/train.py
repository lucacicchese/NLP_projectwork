import argparse
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Folder with images/{train,val_synth} and labels/")
    ap.add_argument("--model", default="yolo26m-pose.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", default="arrows")
    a = ap.parse_args()

    root = Path(a.data_root).resolve()
    yaml_path = root / "arrows.yaml"
    yaml_path.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val_synth\n"
        "kpt_shape: [2, 3]\nflip_idx: [0, 1]\nnames:\n  0: arrow\n  1: double_arrow\n"
    )
    model = YOLO(a.model)
    model.train(
        data=str(yaml_path), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, project=a.project, name=a.name,
        exist_ok=True, workers=16, fliplr=0.0, flipud=0.0, degrees=0.0, mosaic=0.7, close_mosaic=5, scale=0.4,
        hsv_h=0.05, hsv_s=0.5, hsv_v=0.3, patience=100, plots=False,
    )
