"""Bootstrap-SVR (grade ampliada; BCa estrito E com meia-contagem de empates)
+ Conformal-SVR, uma linha JSONL por repeticao. Usa o codigo v0.4 (pasta v2).
Uso: python run_svr_reps2.py <prob> <R> <seed>"""
import sys, json, pathlib, time, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "v2")
import nivel1_benchmark as m, fem_truss as ft
prob, R, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m.N_POOL, m.N_CAND, m.N_INIT = 40_000, 5_000, 12
m.N_BUDGET, m.CONF_NCAL, m.B_BOOT, m.B_AL = 50, 20, 200, 20
m.RNG = np.random.default_rng(seed)
P = {"truss": lambda: ft.TrussFatigue(m.RNG), "4b": lambda: m.FourBranch(6.0),
     "d10": lambda: m.Linear(2.5, 10), "sph": lambda: m.Sphere(3.0, 2),
     "lin2": lambda: m.Linear(2.33, 2), "lin2r": lambda: m.Linear(3.09, 2)}
p = P[prob](); pool, pf = m.build_pool(p)
out = pathlib.Path(f"out/svr2_{prob}_s{seed}.jsonl")
done = sum(1 for _ in out.open()) if out.exists() else 0
m.RNG = np.random.default_rng(seed + 1000 + done)
if prob == "truss": p.rng = m.RNG
for r in range(done, R):
    t = time.time(); rows = []
    for fn in (m.bootstrap_svr, m.conformal_svr):
        for lab, pfh, lo, hi in fn(p, pool)[0]:
            rows.append(dict(method=lab, pf=pfh, lo=lo, hi=hi))
    with out.open("a") as f:
        f.write(json.dumps(dict(r=r, target=pf, rows=rows, sec=round(time.time()-t,1))) + "\n")
