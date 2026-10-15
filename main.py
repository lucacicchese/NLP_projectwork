import argparse
import json
from pathlib import Path

from PIL import Image

import finetune
import inference
import metrics
import visualize


def build_parser():
    parser = argparse.ArgumentParser(description="Qwen3-VL arrow detection pipeline")
    parser.add_argument("--prompt", default=None, help="Override the detection prompt")
    parser.add_argument("--image", default=None, help="Path to a single image")
    parser.add_argument("--dataset", default=None, help="Dataset split dir with images/ and labels/ subdirs")
    parser.add_argument("--zero_shot", action="store_true", help="Run the base model without finetuning")
    parser.add_argument("--train", action="store_true", help="Finetune a LoRA adapter")
    parser.add_argument("--finetuned", action="store_true", help="Run the model with the LoRA adapter")
    parser.add_argument("--adapter_dir", default="adapter", help="LoRA adapter directory (load or save)")
    parser.add_argument("--epochs", type=int, default=1, help="Finetuning epochs for the synthetic-data phase")
    parser.add_argument("--lr", type=float, default=1e-4, help="Finetuning learning rate for the synthetic-data phase")
    parser.add_argument("--real_epochs", type=int, default=1, help="Finetuning epochs for the real-data adaptation phase")
    parser.add_argument("--real_lr", type=float, default=2e-5, help="Finetuning learning rate for the real-data adaptation phase")
    parser.add_argument("--ntl_weight", type=float, default=finetune.NTL_WEIGHT, help="Weight of the Number Token Loss term (0 disables it)")
    parser.add_argument("--max_image_size", type=int, default=384, help="Max image size (px) for resize grid; must match between --train and inference")
    parser.add_argument(
        "--normalized", action="store_true",
        help="Use 0-1000 normalized point coordinates instead of absolute pixels in the resized "
             "image; decouples coordinates from --max_image_size. Must match between --train and inference.",
    )
    parser.add_argument("--output_dir", default="results", help="Where to save predictions and visualizations")
    parser.add_argument("--use_4bit", action="store_true", default=True, help="Load the model in 4-bit")
    parser.add_argument("--device", default="auto", help="Device: auto | cpu | cuda | cuda:0")
    parser.add_argument("--print", "--show", dest="show", action="store_true", help="Visualize predictions")
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)

    if args.train:
        dataset_dir = Path(args.dataset)
        finetune.finetune(
            train_images_dir=dataset_dir / "images",
            train_labels_dir=dataset_dir / "labels",
            adapter_dir=args.adapter_dir,
            epochs=args.epochs,
            lr=args.lr,
            real_epochs=args.real_epochs,
            real_lr=args.real_lr,
            use_4bit=args.use_4bit,
            device=args.device,
            max_image_size=args.max_image_size,
            normalized=args.normalized,
            prompt=args.prompt,
            ntl_weight=args.ntl_weight,
        )
        return

    if args.finetuned:
        model, processor = inference.load_model(
            finetuned=True, adapter_dir=args.adapter_dir, use_4bit=args.use_4bit, device=args.device
        )
    else:
        model, processor = inference.load_model(finetuned=False, use_4bit=args.use_4bit, device=args.device)

    if args.image:
        image_path = Path(args.image)
        arrows = inference.single_image(
            model, processor, str(image_path), prompt=args.prompt, max_image_size=args.max_image_size,
            normalized=args.normalized,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        vis_path = output_dir / f"{image_path.stem}_annotated{image_path.suffix}"
        visualize.draw_boxes(str(image_path), arrows, output_path=str(vis_path), show=args.show)
        print(arrows)
        return

    dataset_dir = Path(args.dataset)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    predictions = inference.full_dataset(
        model, processor, images_dir, output_dir / "json", prompt=args.prompt, max_image_size=args.max_image_size,
        normalized=args.normalized,
    )

    if labels_dir.exists():
        image_sizes = {}
        for stem in predictions:
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    with Image.open(candidate) as im:
                        image_sizes[stem] = im.size
                    break
        report = metrics.evaluate(predictions, labels_dir, image_sizes)
        print(report)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(report, f, indent=2)

    if args.show:
        vis_dir = output_dir / "imgs"
        vis_dir.mkdir(parents=True, exist_ok=True)
        for stem, arrows in predictions.items():
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    gt_boxes = metrics.load_gt_boxes(labels_dir / f"{stem}.csv") if labels_dir.exists() else None
                    visualize.draw_boxes(
                        str(candidate), arrows, gt_boxes, str(vis_dir / f"{stem}_annotated{ext}")
                    )
                    break


if __name__ == "__main__":
    main()
