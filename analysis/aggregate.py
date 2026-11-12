"""Agrega todos os resultados em numbers.json (fonte unica para texto, tabelas e figuras)."""
import json, re, glob, pathlib, numpy as np
from scipy.stats import beta as B, binom, norm, chi2
HERE = pathlib.Path(__file__).resolve().parent; RESULTS = HERE.parent/"results"
SUP = RESULTS/"supplementary"; RES = RESULTS/"earlier_runs"; OLD = RES/"coarse_grid"
NPOOL = 40000
PROB = {"Linear(b=2.33,d=2)": "lin2", "Linear(b=3.09,d=2)": "lin2r", "Linear(b=2.5,d=10)": "d10",
        "Sphere(r=3.0,d=2)": "sph", "FourBranch(k=6.0)": "4b", "TrussFatigue(d=5)": "truss"}
EXACT = {"lin2": norm.cdf(-2.33), "lin2r": norm.cdf(-3.09), "d10": norm.cdf(-2.5),
         "sph": chi2.sf(9, 2), "4b": None, "truss": None}

def parse(path):
    out, cur = {}, None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"PROBLEMA: (.+)", line.strip())
        if m: cur = PROB[m.group(1)]; out[cur] = {"methods": {}}; continue
        m = re.match(r"\s*Pf\(pool N=\d+\) = ([\d.e+-]+)", line)
        if m and cur: out[cur]["pool"] = float(m.group(1)); continue
        m = re.match(r"\s+(\S+)\s+(\d+)%\s+(\d+)%\s+(?:\d+%\s+[\d.]+%\s+)?([\d.e+-]+)\s+([+-][\d.e+-]+)\s+([\d.e+-]+)", line)
        if m and cur:
            out[cur]["methods"][m.group(1)] = dict(cov=int(m.group(2)), up=int(m.group(3)),
                w=float(m.group(4)), bias=float(m.group(5)), IS=float(m.group(6)))
    return out

def metr(rows, t):
    pf = np.array([r["pf"] for r in rows]); lo = np.array([r["lo"] for r in rows]); hi = np.array([r["hi"] for r in rows])
    se = np.sqrt(t*(1-t)/NPOOL); a = 0.05
    miss = hi < t; short = (t-hi[miss])/t
    IS = (hi-lo) + 2/a*np.maximum(lo-t, 0) + 2/a*np.maximum(t-hi, 0)
    return dict(R=len(rows), cov=round(100*np.mean((lo <= t) & (t <= hi))), up=round(100*np.mean(hi >= t)),
                upm=round(100*np.mean(hi >= t-se)), w=float(np.mean(hi-lo)), bias=float(np.mean(pf-t)),
                IS=float(np.mean(IS)), nmiss=int(miss.sum()),
                short_mean=float(short.mean()) if miss.any() else 0.0,
                short_max=float(short.max()) if miss.any() else 0.0,
                miss_pts_max=float(((t-hi[miss])*NPOOL).max()) if miss.any() else 0.0, target=t)

def jl(path): return [json.loads(l) for l in open(path)]

N = {}
fw = parse(RES/"final_wide.txt"); N["fw"] = fw
N["narrow"] = {**parse(OLD/"analytical_R100.txt"), **parse(OLD/"truss_R100.txt")}
N["hiper"] = parse(RES/"controle_hiper.txt")
N["ablK"] = parse(RES/"ablacao_K.txt")
N["exact"] = EXACT
N["se"] = {k: float(np.sqrt(v["pool"]*(1-v["pool"])/NPOOL)) for k, v in fw.items()}
# AK corrigido
N["ak"] = {}
for f in glob.glob(str(SUP/"ak_*_N50_s2025.jsonl")):
    p = re.search(r"ak_(\w+?)_N50", f).group(1); rows = jl(f)
    N["ak"][p] = metr(rows, rows[0]["target"])
rows = jl(SUP/"akard_truss_N50_s2025.jsonl"); N["akard"] = metr(rows, rows[0]["target"])
N["kern"] = {}
for k, f in (("ardrbf", "akardrbf"), ("isomatern", "akisomatern")):
    rows = jl(SUP/f"{f}_truss_N50_s2025.jsonl"); N["kern"][k] = metr(rows, rows[0]["target"])
N["treino30"] = parse(OLD/"controle_treino30.txt")   # replicacao independente, grade estreita
# curva: AK corrigido + BSVR/Conformal da curva fatiada
cf = json.load(open(RES/"curva_fatiada.json")); N["curve"] = {}
for prob in ("truss", "d10"):
    N["curve"][prob] = {}
    for Nb in (30, 50, 80, 120, 150):
        c = cf.get(f"{prob}_N{Nb}", {}).get("metricas", {})
        d = {k: dict(cov=v["cobertura"], w=v["largura"], bias=v["vies"]) for k, v in c.items() if k != "AK-MCS"}
        f = SUP/f"ak_{prob}_N{Nb}_s2025.jsonl"
        if f.exists():
            rr = jl(f); d["AK-MCS"] = metr(rr, rr[0]["target"])
        N["curve"][prob][Nb] = d
# SVR por repeticao (BCa estrito vs meia-contagem, magnitude)
N["svr2"] = {}
for f in glob.glob(str(SUP/"svr2_*_s2025.jsonl")):
    p = re.search(r"svr2_(\w+?)_s2025", f).group(1); reps = jl(f)
    if not reps: continue
    t = reps[0]["target"]; by = {}
    for rp in reps:
        for r in rp["rows"]: by.setdefault(r["method"], []).append(r)
    N["svr2"][p] = {k: metr(v, t) for k, v in by.items()}
    # pareado: estrito erra o limite superior e meia-contagem cobre
    if "BCa-strict" in by:
        s = by["BCa-strict"]; mm = by["BSVR-BCa"]
        N["svr2"][p]["paired_fixed"] = int(sum((a["hi"] < t) and (b["hi"] >= t) for a, b in zip(s, mm)))
        N["svr2"][p]["paired_broken"] = int(sum((a["hi"] >= t) and (b["hi"] < t) for a, b in zip(s, mm)))
# MC-CP exato
def mccp(p, n=50, a=0.05):
    w = c = 0.0
    for k in range(n+1):
        pr = binom.pmf(k, n, p)
        lo = 0 if k == 0 else B.ppf(a/2, k, n-k+1); hi = 1 if k == n else B.ppf(1-a/2, k+1, n-k)
        w += pr*(hi-lo); c += pr*(lo <= p <= hi)
    return dict(cov=round(100*c), w=float(w))
N["mccp"] = {k: mccp(v["pool"]) for k, v in fw.items()}
R = jl(SUP/"ablK_fix.jsonl"); t = R[0]["target"]; N["ablK_fix"] = {}
for K in (500, 2000, 6000):
    lo = np.array([r[f"K{K}"][0] for r in R]); hi = np.array([r[f"K{K}"][1] for r in R])
    N["ablK_fix"][K] = dict(R=len(R), cov=round(100*np.mean((lo <= t) & (t <= hi))), up=round(100*np.mean(hi >= t)))
N["gov"] = dict(total=305, bottom=216, top=87, other=2)   # diag_mech.py, populacao semente 2025
# ================= rodada v2 (run_v2.py, R=100, todos os metodos no mesmo pool) =================
V2 = RESULTS/"main_run"
import collections
reps = [json.loads(l) for l in open(V2/"v2_reps.jsonl")]
byp = collections.defaultdict(lambda: collections.defaultdict(list))
for r in reps: byp[PROB[r["problem"]]][r["method"]].append(r)
def metr2(rows):
    t = rows[0]["target"]; d = metr(rows, t)
    lo = np.array([r["lo"] for r in rows]); hi = np.array([r["hi"] for r in rows]); se = np.sqrt(t*(1-t)/NPOOL)
    d.update(wmed=float(np.median(hi-lo)), nmat=int((hi < t-se).sum()), lowmiss=int((lo > t).sum()),
             bias_med=float(np.median([r["pf"]-t for r in rows])))
    return d
N["v2"] = {p: {m: metr2(rows) for m, rows in d.items()} for p, d in byp.items()}
N["v2_pool"] = {p: d["AK-MCS"][0]["target"] for p, d in byp.items()}
N["v2_se"] = {p: float(np.sqrt(t*(1-t)/NPOOL)) for p, t in N["v2_pool"].items()}
N["v2_pair"] = {}
for p, d in byp.items():
    t = N["v2_pool"][p]; s_, m_ = d["BCa-strict"], d["BSVR-BCa"]
    inn = lambda x: x["lo"] <= t <= x["hi"]
    N["v2_pair"][p] = dict(rescued=int(sum((not inn(a)) and inn(b) for a, b in zip(s_, m_))),
                           broken=int(sum(inn(a) and (not inn(b)) for a, b in zip(s_, m_))))
# colapsos do bootstrap percentil: limite superior abaixo de metade do alvo
N["collapse"] = {}
for p, d in byp.items():
    t = N["v2_pool"][p]
    c = [r for r in d["BSVR-perc"] if r["hi"] < 0.5*t]
    N["collapse"][p] = [dict(r=r["r"], pf=r["pf"], lo=r["lo"], hi=r["hi"], target=t) for r in c]
# largura do conformal vs MC-CP no Four-branch
cw = np.array([r["hi"]-r["lo"] for r in byp["4b"]["Conformal"]])
N["conf4b_frac_wider_mc"] = float(np.mean(cw > N["v2"]["4b"]["MC-CP"]["w"]))
N["mc_zero"] = {p: float(np.mean([r["pf"] == 0 for r in d["MC-CP"]])) for p, d in byp.items()}
# diagnostico de borda
txt = open(V2/"v2.txt", encoding="utf-8").read()
b = {}
b["total"] = int(re.search(r"diagnostico de borda \((\d+) ajustes\)", txt).group(1))
for k in ("C", "gamma", "epsilon"):
    m = re.search(k + r"\s+otimo na borda em\s+(\d+).*?\[min: (\d+), max: (\d+)\]", txt)
    b[k] = dict(edge=int(m.group(1)), min=int(m.group(2)), max=int(m.group(3)))
b["ridge"] = int(re.search(r"simultaneamente: (\d+)", txt).group(1))
N["border"] = b
# conferencia: metricas recalculadas batem com o resumo impresso em v2.txt
chk = parse(V2/"v2.txt")
for p, d in chk.items():
    for m, v in d["methods"].items():
        x = N["v2"][p][m]
        assert v["cov"] == x["cov"] and v["up"] == x["up"], (p, m, v, x["cov"], x["up"])
        assert abs(v["w"]-x["w"]) <= 1e-3*max(abs(v["w"]), 1e-12)+1e-12, (p, m, "w")
print("v2 conferido contra v2.txt:", sum(len(d["methods"]) for d in chk.values()), "linhas")
# ================= controle de C (run_controle_C.py, R=50 por braco, bracos pareados) =================
CC = RESULTS/"control_C"; prog = json.load(open(CC/"progresso.json")); N["ctrlC"] = {}
for p in ("truss", "4b", "d10"):
    arms = {}
    for arm in ("base", "wide"):
        by = collections.defaultdict(list)
        for l in open(CC/f"reps_{p}_{arm}.jsonl"):
            r = json.loads(l); by[r["method"]].append(r)
        arms[arm] = by
    t = arms["base"]["BSVR-perc"][0]["target"]; se = np.sqrt(t*(1-t)/NPOOL)
    assert all(r["target"] == t for a in arms.values() for rows in a.values() for r in rows), p   # mesma populacao
    out = dict(target=t, pair={})
    for arm, by in arms.items():
        out[arm] = {}
        for m, rows in by.items():
            assert [r["r"] for r in rows] == list(range(len(rows))), (p, arm, m)
            lo = np.array([r["lo"] for r in rows]); hi = np.array([r["hi"] for r in rows]); miss = hi < t
            out[arm][m] = dict(R=len(rows), cov=round(100*np.mean((lo <= t) & (t <= hi))), up=round(100*np.mean(hi >= t)),
                               upm=round(100*np.mean(hi >= t-se)), nmiss=int(miss.sum()), nmat=int((hi < t-se).sum()),
                               short_max=float(((t-hi[miss])/t).max()) if miss.any() else 0.0,
                               w=float(np.mean(hi-lo)), wmed=float(np.median(hi-lo)),
                               nhalf=int((hi < 0.5*t).sum()), zero=[int(r["r"]) for r in rows if r["hi"] == 0])
        c = prog[f"{p}_{arm}"]["counts"]; n = c["total"]
        out[arm]["fits"] = dict(total=n, big=sum(v for k, v in c.items() if k in ("C=1e+06", "C=1e+07"))/n,
                                at1e5=c.get("C=1e+05", 0)/n, cap=c.get("bateu_teto_iter", 0)/n)
    inn = lambda r: r["lo"] <= t <= r["hi"]
    for m in ("BSVR-perc", "BSVR-BCa", "Conformal"):
        b, w_ = arms["base"][m], arms["wide"][m]
        out["pair"][m] = dict(gained=int(sum((not inn(x)) and inn(y) for x, y in zip(b, w_))),
                              lost=int(sum(inn(x) and (not inn(y)) for x, y in zip(b, w_))))
    # repeticoes com colapso total num braco: estimativa pontual no outro braco
    out["zero_other"] = {arm: [dict(r=r, pf_other=arms[o]["BSVR-perc"][r]["pf"]) for r in out[arm]["BSVR-perc"]["zero"]]
                         for arm, o in (("base", "wide"), ("wide", "base"))}
    N["ctrlC"][p] = out
json.dump(N, open(HERE/"numbers.json", "w"), indent=1, default=float)
print("ok; AK:", sorted(N["ak"]), "svr2:", {k: v.get("BSVR-perc", {}).get("R") for k, v in N["svr2"].items()})
