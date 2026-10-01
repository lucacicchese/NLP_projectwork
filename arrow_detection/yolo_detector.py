"""YOLO-based arrow detector used as the reference baseline."""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from .utils import select_best_cuda_device

logger = logging.getLogger(__name__)

ARROW_CLASS_NAMES = {"arrow", "arrows", "arrowhead", "line", "connection", "connector"}

DEFAULT_YOLO_MODEL = "best.pt"


def _load_model(model_path: str, device: str):
    """Load a YOLOv5 model via torch.hub."""
    logger.info(f"Loading YOLOv5 model: {model_path} on {device}")
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=model_path,
        force_reload=False,
    )
    model.to(device)
    return model


class YOLOArrowDetector:
    """Detect arrows with a YOLOv5 model and save results in x1 y1 x2 y2 CSV format."""

    def __init__(
        self,
        model_id: str = DEFAULT_YOLO_MODEL,
        device: str = "auto",
        conf_threshold: float = 0.25,
        arrow_classes: Optional[List[str]] = None,
    ):
        if device == "auto":
            self.device = select_best_cuda_device()
        elif device == "cuda" and torch.cuda.is_available():
            self.device = select_best_cuda_device()
        else:
            self.device = device

        self.conf_threshold = conf_threshold
        self.arrow_class_names = set(arrow_classes) if arrow_classes else ARROW_CLASS_NAMES

        self.model = _load_model(model_id, self.device)
        self.model.conf = conf_threshold

        self.class_names: Dict[int, str] = self.model.names
        self.arrow_class_ids = self._resolve_arrow_ids()

        if self.arrow_class_ids:
            names = [self.class_names[i] for i in self.arrow_class_ids]
            logger.info(f"Arrow classes found in model: {names} (ids={self.arrow_class_ids})")
        else:
            logger.warning(
                "No arrow-related class found in this model. "
                f"Model classes: {list(self.class_names.values())}. "
                "All detections will be returned. Set arrow_classes=['<your_class>'] to filter."
            )

    def _resolve_arrow_ids(self) -> List[int]:
        ids = []
        for cls_id, cls_name in self.class_names.items():
            if cls_name.lower() in self.arrow_class_names:
                ids.append(cls_id)
        return ids

    def detect(self, image_path: str) -> List[Dict[str, int]]:
        """Return list of {x1, y1, x2, y2} dicts in pixel coordinates."""
        results = self.model(image_path)
        # xyxy[0] shape: [N, 6] — x1, y1, x2, y2, conf, cls
        detections = []
        for *xyxy, conf, cls_id in results.xyxy[0].tolist():
            cls_id = int(cls_id)
            if self.arrow_class_ids and cls_id not in self.arrow_class_ids:
                continue
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return detections

    def run_detection(
        self,
        input_dir: str,
        output_dir: str,
        image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    ):
        """Process all images in input_dir and write CSV + annotated images."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        csv_dir = output_path / "csv"
        imgs_dir = output_path / "imgs"
        csv_dir.mkdir(parents=True, exist_ok=True)
        imgs_dir.mkdir(parents=True, exist_ok=True)

        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        image_files = sorted(set(image_files))

        logger.info(f"Found {len(image_files)} images in {input_dir}")

        for img_path in tqdm(image_files, desc="YOLO detection"):
            try:
                detections = self.detect(str(img_path))
                self._save_csv(detections, csv_dir / f"{img_path.stem}.csv")
                self._save_annotated(
                    str(img_path),
                    detections,
                    imgs_dir / f"{img_path.stem}_yolo{img_path.suffix}",
                )
                logger.debug(f"{img_path.name}: {len(detections)} arrows")
            except Exception as e:
                logger.error(f"Error on {img_path.name}: {e}")

        logger.info(f"YOLO results saved to {output_dir}")

    @staticmethod
    def _save_csv(detections: List[Dict[str, int]], path: Path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["x1", "y1", "x2", "y2"])
            writer.writeheader()
            writer.writerows(detections)

    @staticmethod
    def _save_annotated(image_path: str, detections: List[Dict[str, int]], output_path: Path):
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        for det in detections:
            x1 = max(0, min(det["x1"], w - 1))
            y1 = max(0, min(det["y1"], h - 1))
            x2 = max(0, min(det["x2"], w - 1))
            y2 = max(0, min(det["y2"], h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            draw.rectangle([(x1, y1), (x2, y2)], outline="#2196F3", width=3)

        img.save(output_path)
