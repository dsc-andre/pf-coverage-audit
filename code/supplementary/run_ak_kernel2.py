"""Controle de kernel na trelica, separando os dois fatores do controle anterior
(ARD Matern-5/2): anisotropia e suavidade.
Uso: python run_ak_kernel2.py <ardrbf|isomatern> <R> <seed>"""
import sys, json, pathlib, numpy as np, warnings; warnings.filterwarnings("ignore")
import nivel1_benchmark as m, fem_truss as ft, ak_fix
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as Ck, Matern, RBF
kind, R, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m.N_POOL, m.N_INIT, m.M_POST, m.AK_SIM_K, m.N_BUDGET = 40_000, 12, 200, 2000, 50
m.RNG = np.random.default_rng(seed)
p = ft.TrussFatigue(m.RNG); pool, pf = m.build_pool(p)
K = {"ardrbf": lambda: RBF(np.ones(p.d), (1e-3, 1e3)),
     "isomatern": lambda: Matern(1.0, (1e-3, 1e3), nu=2.5)}[kind]
m.make_gp = lambda: GaussianProcessRegressor(Ck(1.0, (1e-5, 1e8))*K(), normalize_y=True,
                                             n_restarts_optimizer=1, alpha=1e-10)
out = pathlib.Path(f"out/ak{kind}_truss_N50_s{seed}.jsonl")
done = sum(1 for _ in out.open()) if out.exists() else 0
for r in range(R):
    st = []; (lab, pfh, lo, hi), = ak_fix.ak_mcs_fixed(p, pool, st)[0]
    if r < done: continue
    with out.open("a") as f:
        f.write(json.dumps(dict(r=r, pf=pfh, lo=lo, hi=hi, target=pf, n_unc=st[0]["n_unc"], kernel=st[0]["kernel"])) + "\n")
