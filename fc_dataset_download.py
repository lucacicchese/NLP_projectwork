import io
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image


DATASET_NAME = "galirage/FC-Detection"
OUTPUT_DIR = "training_data"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main"

SPLITS = {
    "train": ("data_coco_format_start_end_train", "train"),
    "val": ("data_coco_format_start_end_test", "val"),
}


def fetch_json(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def fetch_bytes(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.content


def save_split(remote_dir, images_dir, labels_dir):
    print(f"\nProcessing split: {remote_dir}")

    annotations_url = f"{BASE_URL}/{remote_dir}/annotations/instances_custom.json"
    coco = fetch_json(annotations_url)

    categories = {c["id"]: c["name"] for c in coco["categories"]}

    boxes_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        boxes_by_image[ann["image_id"]].append(
            (x, y, x + w, y + h, categories[ann["category_id"]])
        )

    images = coco["images"]

    for idx, image_info in enumerate(images):
        file_name = image_info["file_name"]
        image_url = f"{BASE_URL}/{remote_dir}/images/{file_name}"

        image = Image.open(io.BytesIO(fetch_bytes(image_url)))
        if image.mode != "RGB":
            image = image.convert("RGB")

        filename = Path(file_name).stem + ".jpg"
        image.save(images_dir / filename, format="JPEG")

        csv_path = labels_dir / (Path(filename).stem + ".csv")
        with open(csv_path, "w") as f:
            f.write("xmin,ymin,xmax,ymax,label\n")
            for xmin, ymin, xmax, ymax, label in boxes_by_image[image_info["id"]]:
                f.write(f"{xmin},{ymin},{xmax},{ymax},{label}\n")

        print(f"[{idx + 1}/{len(images)}] Saved: {filename}")


def main():
    for remote_dir, local_split in SPLITS.values():
        images_dir = Path(OUTPUT_DIR) / local_split / "images"
        labels_dir = Path(OUTPUT_DIR) / local_split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        save_split(remote_dir, images_dir, labels_dir)

    print("\nDone!")
    print(f"Dataset saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
