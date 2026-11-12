import numpy as np, warnings; warnings.filterwarnings("ignore")
import nivel1_benchmark as m, fem_truss as ft, ak_fix
m.N_POOL=40000; m.N_INIT=12; m.M_POST=200; m.AK_SIM_K=2000; m.N_BUDGET=50
m.RNG=np.random.default_rng(2025); p=ft.TrussFatigue(m.RNG)
pool,pf=m.build_pool(p); g=p.g(pool)
def gov(U):
    U=np.atleast_2d(U); out=[]
    for u in U:
        P=p._ln(u[0],p.P_m,p.P_cv); A=np.array([p._ln(u[1],p.Ac_m,p.Ac_cv),p._ln(u[2],p.Ad_m,p.Ad_cv),p._ln(u[3],p.Av_m,p.Av_cv)])
        # recompute member stresses
        Am=A[ft.GROUP]; ndof=12; K=np.zeros((ndof,ndof))
        for k,(i,j) in enumerate(ft.MEMBERS):
            c,s=ft._COS[k]; kk=ft.E_FIX*Am[k]/ft._L[k]
            ke=kk*np.array([[c*c,c*s,-c*c,-c*s],[c*s,s*s,-c*s,-s*s],[-c*c,-c*s,c*c,c*s],[-c*s,-s*s,c*s,s*s]])
            d=[2*i,2*i+1,2*j,2*j+1]; K[np.ix_(d,d)]+=ke
        F=np.zeros(ndof); F[ft.LOAD_DOFS]=-P; free=[d for d in range(ndof) if d not in ft.FIXED]
        uu=np.zeros(ndof); uu[free]=np.linalg.solve(K[np.ix_(free,free)],F[free])
        st=[abs(ft.E_FIX*((uu[2*j]-uu[2*i])*ft._COS[k][0]+(uu[2*j+1]-uu[2*i+1])*ft._COS[k][1])/ft._L[k]) for k,(i,j) in enumerate(ft.MEMBERS)]
        out.append(int(np.argmax(st)))
    return np.array(out)
fail=np.where(g<=0)[0]; gv=gov(pool[fail]); print("pf",pf,"governing member among failures:",np.bincount(gv,minlength=10))
for rep in range(4):
    Xd=p.sample(12); yd=p.g(Xd); used=np.zeros(len(pool),bool)
    for n in range(12,50):
        gp=m.make_gp().fit(Xd,yd); mu,sd=gp.predict(pool,return_std=True); sd=np.maximum(sd,1e-12)
        U=np.abs(mu)/sd; U[used]=np.inf; i=int(np.argmin(U)); used[i]=True
        Xd=np.vstack([Xd,pool[i]]); yd=np.append(yd,g[i])
    gp=m.make_gp().fit(Xd,yd); mu,sd=gp.predict(pool,return_std=True)
    mis=np.where((mu<=0)!=(g<=0))[0]
    print(f"rep{rep} kernel {gp.kernel_} misclassified {len(mis)} | U<2: {np.sum(np.abs(mu)/sd<2)}")
    for j in mis: print(f"   g={g[j]:+.4f} mu={mu[j]:+.4f} sd={sd[j]:.2e} U={abs(mu[j])/sd[j]:.1f} |u|={np.linalg.norm(pool[j]):.2f} gov={gov(pool[j])[0]}")
