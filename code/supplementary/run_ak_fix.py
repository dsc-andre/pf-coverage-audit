"""Re-roda SO o AK-MCS (versao corrigida: sem re-selecao de pontos ja avaliados)
e grava cada repeticao em JSONL (pf, lo, hi, alvo), para metricas de magnitude.
Uso: python run_ak_fix.py <truss|d10> <N_budget> <R> <seed>"""
import sys, json, time, pathlib, numpy as np, warnings
warnings.filterwarnings("ignore")
import nivel1_benchmark as m, fem_truss as ft, ak_fix
prob, N, R, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
m.N_POOL, m.N_CAND, m.N_INIT = 40_000, 5_000, 12
m.M_POST, m.AK_SIM_K, m.N_BUDGET = 200, 2000, N
m.RNG = np.random.default_rng(seed)
P = {"truss": lambda: ft.TrussFatigue(m.RNG), "d10": lambda: m.Linear(2.5, 10), "4b": lambda: m.FourBranch(6.0),
     "sph": lambda: m.Sphere(3.0, 2), "lin2": lambda: m.Linear(2.33, 2), "lin2r": lambda: m.Linear(3.09, 2)}
p = P[prob]()
pool, pf = m.build_pool(p)
out = pathlib.Path(f"out/ak_{prob}_N{N}_s{seed}.jsonl"); out.parent.mkdir(exist_ok=True)
done = sum(1 for _ in out.open()) if out.exists() else 0
for r in range(R):
    st = []
    (lab, pfh, lo, hi), = ak_fix.ak_mcs_fixed(p, pool, st)[0]
    if r < done: continue          # retomada: consome o RNG igual, nao regrava
    with out.open("a") as f:
        f.write(json.dumps(dict(r=r, pf=pfh, lo=lo, hi=hi, target=pf,
                                n_unc=st[0]["n_unc"], kernel=st[0]["kernel"])) + "\n")
