"""Ablacao sobre K com o AK-MCS corrigido: mesmo laco adaptativo, tres valores de K."""
import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
import nivel1_benchmark as m, fem_truss as ft
R = int(sys.argv[1]); m.N_POOL, m.N_INIT, m.M_POST, m.N_BUDGET = 40000, 12, 200, 50
m.RNG = np.random.default_rng(2025); p = ft.TrussFatigue(m.RNG); pool, pf = m.build_pool(p)
for r in range(R):
    rng = np.random.default_rng(5000+r); p.rng = rng
    Xd = p.sample(12); yd = p.g(Xd); used = np.zeros(len(pool), bool)
    for n in range(12, 50):
        gp = m.make_gp().fit(Xd, yd); mu, sd = gp.predict(pool, return_std=True); sd = np.maximum(sd, 1e-12)
        U = np.abs(mu)/sd; U[used] = np.inf; i = int(np.argmin(U)); used[i] = True
        Xd = np.vstack([Xd, pool[i]]); yd = np.append(yd, p.g(pool[i:i+1]))
    gp = m.make_gp().fit(Xd, yd); mu, sd = gp.predict(pool, return_std=True); sd = np.maximum(sd, 1e-12)
    pfh = float(np.mean(mu <= 0)); U = np.abs(mu)/sd; rec = dict(r=r, pf=pfh, target=pf)
    for K in (500, 2000, 6000):
        idx = np.argsort(U)[:K]; mu_s, cov_s = gp.predict(pool[idx], return_cov=True); jit = 1e-8
        for _ in range(6):
            try: L = np.linalg.cholesky(cov_s + jit*np.eye(K)); break
            except np.linalg.LinAlgError: jit *= 10
        else: L = np.diag(np.sqrt(np.maximum(np.diag(cov_s), 1e-12)))
        ff = int(np.sum(mu <= 0) - np.sum(mu[idx] <= 0)); g2 = np.random.default_rng(9000+r)
        pr = [(ff + int(np.sum(mu_s + L @ g2.standard_normal(K) <= 0)))/len(pool) for _ in range(200)]
        rec[f"K{K}"] = [float(np.percentile(pr, 2.5)), float(np.percentile(pr, 97.5))]
    open("out/ablK_fix.jsonl", "a").write(json.dumps(rec) + "\n")
