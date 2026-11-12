import sys, json, pathlib, numpy as np, warnings; warnings.filterwarnings("ignore")
import nivel1_benchmark as m, fem_truss as ft, ak_fix
prob, N, R, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
m.N_POOL, m.N_INIT, m.M_POST, m.AK_SIM_K, m.N_BUDGET = 40_000, 12, 200, 2000, N
m.RNG = np.random.default_rng(seed)
p = ft.TrussFatigue(m.RNG) if prob == "truss" else (m.FourBranch(6.0) if prob=="4b" else m.Sphere(3.0,2))
pool, pf = m.build_pool(p)
m.make_gp = ak_fix.make_gp_ard(p.d)
out = pathlib.Path(f"out/akard_{prob}_N{N}_s{seed}.jsonl")
for r in range(R):
    st=[]; (lab,pfh,lo,hi), = ak_fix.ak_mcs_fixed(p,pool,st)[0]
    with out.open("a") as f: f.write(json.dumps(dict(r=r,pf=pfh,lo=lo,hi=hi,target=pf,n_unc=st[0]["n_unc"],kernel=st[0]["kernel"]))+"\n")
