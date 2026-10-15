import argparse
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

import metrics
from inference import IMAGE_EXTENSIONS, PROMPT, PROMPT_NORMALIZED, load_model, single_image
from self_consistency import aggregate, consensus_arrows, sample_arrows

MIN_FRACS = (0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
METRICS = ("precision", "recall", "f1", "mean_dist_pct_diag")


def summary(values):
    if not values:
        return None
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values), "min": min(values), "max": max(values)}


def agreement_f1(a, b, tol):
    if not a and not b:
        return 1.0
    available = list(b)
    matched = 0
    for arrow in a:
        for j, other in enumerate(available):
            if math.dist(arrow["tail"], other["tail"]) <= tol and math.dist(arrow["head"], other["head"]) <= tol:
                matched += 1
                available.pop(j)
                break
    return 2 * matched / (len(a) + len(b))


def spread(points, diag):
    center = (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
    return 100 * statistics.mean(math.dist(p, center) for p in points) / diag


def analyze(samples, greedy, sizes, labels_dir, n_samples, tol_frac):
    stems = sorted(samples)
    tols = {s: tol_frac * math.hypot(*sizes[s]) for s in stems}

    def evaluate(predictions):
        return metrics.evaluate(predictions, labels_dir, sizes)

    runs = [evaluate({s: samples[s][k] for s in stems}) for k in range(n_samples)]
    counts = [[len(arrows) for arrows in samples[s]] for s in stems]
    agreements = [
        agreement_f1(samples[s][i], samples[s][j], tols[s])
        for s in stems
        for i, j in itertools.combinations(range(n_samples), 2)
    ]

    support = Counter()
    spreads = {"tail": [], "head": []}
    for s in stems:
        for c in aggregate(samples[s], tols[s]):
            support[len(c["samples"])] += 1
            if len(c["tails"]) > 1:
                spreads["tail"].append(spread(c["tails"], math.hypot(*sizes[s])))
                spreads["head"].append(spread(c["heads"], math.hypot(*sizes[s])))

    sweep = [
        {"n_samples": n, "min_frac": f, **evaluate({s: consensus_arrows(aggregate(samples[s][:n], tols[s]), n, f) for s in stems})}
        for n in range(1, n_samples + 1)
        for f in MIN_FRACS
    ]

    return {
        "n_images": len(stems),
        "n_samples": n_samples,
        "tol_frac": tol_frac,
        "greedy": evaluate(greedy),
        "single_sample_runs": {key: summary([r[key] for r in runs]) for key in METRICS},
        "pairwise_sample_agreement_f1": summary(agreements),
        "arrow_count_std_per_image": summary([statistics.pstdev(c) for c in counts]),
        "frac_images_unanimous_count": sum(len(set(c)) == 1 for c in counts) / len(stems),
        "cluster_support_distribution": {str(k): support[k] / sum(support.values()) for k in range(1, n_samples + 1)},
        "cluster_point_spread_pct_diag": {key: summary(v) for key, v in spreads.items()},
        "consensus_sweep": sweep,
    }


def fmt(r):
    return f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} dist={r['mean_dist_pct_diag']:.2f}%"


def print_report(report):
    print(f"\nGreedy: {fmt(report['greedy'])}")
    print("Single sampled runs (mean ± std over runs):")
    for key, v in report["single_sample_runs"].items():
        print(f"  {key:20s} {v['mean']:.3f} ± {v['std']:.3f}  [{v['min']:.3f}, {v['max']:.3f}]")
    print(f"Pairwise sample agreement F1: {report['pairwise_sample_agreement_f1']['mean']:.3f}")
    print(f"Arrow count std per image: {report['arrow_count_std_per_image']['mean']:.2f}")
    print(f"Images where all samples have the same arrow count: {report['frac_images_unanimous_count']:.2f}")
    spreads = report["cluster_point_spread_pct_diag"]
    if spreads["tail"]:
        print(f"Cluster spread (% diag): tail {spreads['tail']['mean']:.2f}  head {spreads['head']['mean']:.2f}")
    print(f"Cluster support distribution: {report['cluster_support_distribution']}")
    print("\nConsensus sweep:")
    print(f"  {'N':>3} {'min_frac':>8} {'P':>6} {'R':>6} {'F1':>6} {'dist%':>6}")
    for r in report["consensus_sweep"]:
        print(f"  {r['n_samples']:>3} {r['min_frac']:>8.1f} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {r['mean_dist_pct_diag']:>6.2f}")


def main():
    parser = argparse.ArgumentParser(description="Variance analysis of self-consistency sampling")
    parser.add_argument("--dataset", default="training_data/val")
    parser.add_argument("--adapter_dir", default="adapter_normalized_v4")
    parser.add_argument("--output_dir", default="results/sc_variance")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tol_frac", type=float, default=0.03)
    parser.add_argument("--max_image_size", type=int, default=384)
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.json"
    data = json.loads(samples_path.read_text()) if samples_path.exists() else {"samples": {}, "greedy": {}, "sizes": {}}

    prompt = PROMPT_NORMALIZED if args.normalized else PROMPT
    images = sorted(p for p in (Path(args.dataset) / "images").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    todo = [p for p in images if len(data["samples"].get(p.stem, [])) < args.n_samples]
    if todo:
        model, processor = load_model(finetuned=True, adapter_dir=args.adapter_dir, device=args.device)
        for path in tqdm(todo, desc="Sampling"):
            with Image.open(path) as image:
                data["sizes"][path.stem] = image.size
            data["greedy"][path.stem] = single_image(
                model, processor, str(path), prompt, args.max_image_size, normalized=args.normalized
            )
            data["samples"][path.stem] = sample_arrows(
                model, processor, path, prompt, args.max_image_size, args.normalized, args.n_samples, args.temperature
            )
            samples_path.write_text(json.dumps(data))

    samples = {s: v[: args.n_samples] for s, v in data["samples"].items()}
    sizes = {s: tuple(v) for s, v in data["sizes"].items()}
    report = analyze(samples, data["greedy"], sizes, Path(args.dataset) / "labels", args.n_samples, args.tol_frac)
    (output_dir / "variance_report.json").write_text(json.dumps(report, indent=2))
    print_report(report)


if __name__ == "__main__":
    main()
