import json
import sys
from pathlib import Path

print(f"{'run':45s} {'P':>6} {'R':>6} {'F1':>6} {'dist%':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
for d in sys.argv[1:]:
    path = Path(d) / "metrics.json"
    if not path.exists():
        print(f"{d:45s} missing")
        continue
    r = json.loads(path.read_text())
    print(f"{d:45s} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {r['mean_dist_pct_diag']:>6.2f} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5}")
