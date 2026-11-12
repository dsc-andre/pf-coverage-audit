import sys, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import nivel1_benchmark as m, fem_truss as ft, ak_fix
prob_name=sys.argv[1]; N=int(sys.argv[2]); R=int(sys.argv[3])
m.N_POOL=40000; m.N_INIT=12; m.M_POST=200; m.AK_SIM_K=2000; m.N_BUDGET=N
m.RNG=np.random.default_rng(int(sys.argv[4]) if len(sys.argv)>4 else 777)
p = ft.TrussFatigue(m.RNG) if prob_name=="truss" else m.Linear(beta=2.5,d=10)
pool,pf=m.build_pool(p); print("pf pool",pf, flush=True)
# count duplicates in original by monkeypatching sample of g calls
res={"orig":[], "fix":[]}; dups=0; stats=[]
for r in range(R):
    seed=1000+r
    m.RNG=np.random.default_rng(seed); p.rng=m.RNG if prob_name=="truss" else None
    (lab,pfh,lo,hi),=m.ak_mcs(p,pool)[0]
    res["orig"].append((pfh,lo,hi))
    m.RNG=np.random.default_rng(seed)
    if prob_name=="truss": p.rng=m.RNG
    (lab,pfh2,lo2,hi2),=ak_fix.ak_mcs_fixed(p,pool,stats)[0]
    res["fix"].append((pfh2,lo2,hi2))
    print(f"r{r} orig {pfh:.4e} [{lo:.4e},{hi:.4e}] | fix {pfh2:.4e} [{lo2:.4e},{hi2:.4e}] {stats[-1]}",flush=True)
for k,v in res.items():
    v=np.array(v); cov=np.mean((v[:,1]<=pf)&(pf<=v[:,2])); up=np.mean(v[:,2]>=pf)
    print(k,"cov",cov,"up",up,"width",np.mean(v[:,2]-v[:,1]),"bias",np.mean(v[:,0]-pf))
