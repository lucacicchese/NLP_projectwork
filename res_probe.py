import argparse
import gc
import json
import time
from pathlib import Path

import torch

import box_check
from box_eval import evaluate
from inference import full_dataset, load_model

LOOP = ["104", "132", "144", "145", "149", "167", "169", "234", "269", "296"]
OK = ["14", "105", "28", "241", "178", "13", "245", "218", "252", "309"]


def subset(test_dir, stems, dst):
    for kind in ("images", "labels"):
        (dst / kind).mkdir(parents=True, exist_ok=True)
    for stem in stems:
        for kind in ("images", "labels"):
            for src in (test_dir / kind).glob(f"{stem}.*"):
                link = dst / kind / src.name
                if not link.exists():
                    link.symlink_to(src.resolve())
    return dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="dataset_unifi_split_clean/test")
    parser.add_argument("--adapters", nargs="+", default=["adapter_normalized_v4", "adapter_unifi_box"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[384, 768])
    parser.add_argument("--out", default="results/res_probe")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out = Path(args.out)
    groups = {name: subset(Path(args.test), stems, out / f"subset_{name}") for name, stems in (("loop", LOOP), ("ok", OK))}
    summary = []
    for adapter in args.adapters:
        model, processor = load_model(finetuned=True, adapter_dir=adapter, device=args.device)
        for size in args.sizes:
            for group, data in groups.items():
                run = out / f"{Path(adapter).name}_{size}" / group
                start = time.time()
                preds = full_dataset(model, processor, data / "images", run / "json", max_image_size=size,
                                     normalized=True, save_raw=True)
                seconds = (time.time() - start) / len(preds)
                res = evaluate(preds, data / "images", data / "labels", box_check.MARGIN_FRAC, box_check.INSIDE_THRESHOLD)
                raws = [json.loads(p.read_text()) for p in (run / "raw").glob("*.json")]
                row = {
                    "adapter": Path(adapter).name, "size": size, "group": group,
                    "box_f1": res["box"]["f1"], "box_r": res["box"]["recall"], "pck_f1": res["pck"]["f1"],
                    "loops": sum(r["loop_stopped"] for r in raws), "hit_limit": sum(r["hit_limit"] for r in raws),
                    "zero_arrows": sum(r["parsed_arrows"] == 0 for r in raws), "s_per_img": round(seconds, 1),
                }
                summary.append(row)
                print(json.dumps(row), flush=True)
                (out / "summary.json").write_text(json.dumps(summary, indent=2))
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'adapter':24s} {'size':>4s} {'group':5s} {'boxF1':>6s} {'boxR':>6s} {'pckF1':>6s} {'loops':>5s} {'empty':>5s} {'s/img':>6s}")
    for r in summary:
        print(f"{r['adapter']:24s} {r['size']:4d} {r['group']:5s} {r['box_f1']:6.3f} {r['box_r']:6.3f} {r['pck_f1']:6.3f} "
              f"{r['loops']:5d} {r['zero_arrows']:5d} {r['s_per_img']:6.1f}")


if __name__ == "__main__":
    main()
