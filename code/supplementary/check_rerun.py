"""Compares a rerun of run_ak_fix.py with the stored results.

Usage (from this folder, after `python run_ak_fix.py truss 50 100 2025`):
    python check_rerun.py truss 50 2025

Reads   out/ak_<problem>_N<budget>_s<seed>.jsonl                      (the rerun)
and     ../../results/supplementary/ak_<problem>_N<budget>_s<seed>.jsonl  (stored)

Populations, designs and conditional simulations are seeded, so the target and the
sequence of designs are the same. The restart of the Gaussian-process hyperparameter
optimiser is not seeded, so individual repetitions can differ by a few population
points; the summary figures should agree within a few percentage points.
"""
import json, pathlib, sys
import numpy as np

prob, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
HERE = pathlib.Path(__file__).resolve().parent
name = f"ak_{prob}_N{N}_s{seed}.jsonl"
new = [json.loads(l) for l in open(HERE / "out" / name)]
old = [json.loads(l) for l in open(HERE.parent.parent / "results" / "supplementary" / name)][:len(new)]
NPOOL = 40_000
t = new[0]["target"]; se = np.sqrt(t * (1 - t) / NPOOL)

print(f"repetitions compared: {len(new)}")
print(f"same target (population): {t == old[0]['target']}  (P_f* = {t:.4e})")
same = sum((a["pf"], a["lo"], a["hi"]) == (b["pf"], b["lo"], b["hi"]) for a, b in zip(new, old))
print(f"identical repetitions: {same} of {len(new)}")
d = max(abs(a["hi"] - b["hi"]) for a, b in zip(new, old)) * NPOOL
print(f"largest difference in the upper bound: {d:.1f} population points")
print(f"{'':8s} {'cov':>5s} {'upper':>6s} {'material':>9s} {'max shortfall (points)':>23s} {'mean width':>11s}")
for label, rows in (("rerun", new), ("stored", old)):
    lo = np.array([r["lo"] for r in rows]); hi = np.array([r["hi"] for r in rows]); miss = hi < t
    short = float(((t - hi[miss]) * NPOOL).max()) if miss.any() else 0.0
    print(f"{label:8s} {100*np.mean((lo <= t) & (t <= hi)):4.0f}% {100*np.mean(hi >= t):5.0f}% "
          f"{100*np.mean(hi >= t - se):8.0f}% {short:23.1f} {np.mean(hi - lo):11.2e}")
