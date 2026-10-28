import argparse
import json
from pathlib import Path

from PIL import Image

import box_check
import metrics
from inference import full_dataset, load_model


def evaluate(predictions, images_dir, labels_dir, margin, threshold):
    sizes = {}
    for image_path in Path(images_dir).iterdir():
        if image_path.stem in predictions:
            with Image.open(image_path) as im:
                sizes[image_path.stem] = im.size
    return {
        "pck": metrics.evaluate(predictions, labels_dir, sizes),
        "box": box_check.evaluate_box(predictions, labels_dir, sizes, margin, threshold),
    }


def main():
    parser = argparse.ArgumentParser(description="Inference with PCK and GT-box metrics")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--margin", type=float, default=box_check.MARGIN_FRAC)
    parser.add_argument("--threshold", type=float, default=box_check.INSIDE_THRESHOLD)
    parser.add_argument("--max_image_size", type=int, default=384)
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    dataset, output_dir = Path(args.dataset), Path(args.output_dir)
    model, processor = load_model(finetuned=True, adapter_dir=args.adapter_dir, device=args.device)
    predictions = full_dataset(
        model, processor, dataset / "images", output_dir / "json", max_image_size=args.max_image_size,
        max_new_tokens=args.max_new_tokens, normalized=True, save_raw=True,
    )
    result = evaluate(predictions, dataset / "images", dataset / "labels", args.margin, args.threshold)
    raws = [json.loads(p.read_text()) for p in (output_dir / "raw").glob("*.json")]
    result["outputs"] = {
        "images": len(raws),
        "hit_token_limit": sum(r["hit_limit"] for r in raws),
        "loop_stopped": sum(r.get("loop_stopped", False) for r in raws),
        "empty": sum(r["parsed_arrows"] == 0 for r in raws),
    }

    for name in ("pck", "box"):
        r = result[name]
        print(f"{name:4s} P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1']:.3f}  TP {r['tp']}  FP {r['fp']}  FN {r['fn']}")
    print(f"arrows inside a box (one per box): {result['box']['tp']}, with the right direction: {result['box']['direction_ok']}")
    print(f"outputs: {result['outputs']['hit_token_limit']} of {result['outputs']['images']} hit max_new_tokens, "
          f"{result['outputs']['loop_stopped']} cut as loops, {result['outputs']['empty']} without any arrow")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
