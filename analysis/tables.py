"""Builds the tables of the paper from numbers.json and writes tables.md.

Run aggregate.py first. Table 2 (random variables of the truss) is input data, defined in code/main/fem_truss.py.
"""
import json, re, pathlib, sys
H = pathlib.Path(__file__).resolve().parent
N = json.load(open(H/"numbers.json"))
v, pool, se = N["v2"], N["v2_pool"], N["v2_se"]
def _sup(n): return str(n).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
def e(x, d=1):
    if abs(x) < 1e-12: return "0"      # zero ou ruido de ponto flutuante
    m, ex = f"{x:.{d}e}".split("e")
    return f"{m}×10{_sup(int(ex))}"
def sgn(x, d=1): return "0" if abs(x) < 1e-12 else ("+" if x > 0 else "−") + e(abs(x), d)
def pct(x): return f"{x:.0f}%"
def pr(x, d=1): return f"{100*x:.{d}f}%"
def shf(x): return pr(x, 1) if x < 0.0995 else pr(x, 0)        # falta relativa: 1 casa abaixo de 10%
NAME = {"lin2": "Linear, d=2", "lin2r": "Linear, d=2 (rare)", "d10": "Linear, d=10", "sph": "Sphere, d=2",
        "4b": "Four-branch, d=2", "truss": "Truss (FE), d=5"}
P = ["lin2", "lin2r", "d10", "sph", "4b"]
MAIN = [("AK-MCS", ""), ("BSVR-perc", ""), ("BSVR-BCa", ""), ("Conformal", ""), ("MC-CP", "")]
NB = lambda t: t.replace("-", "\u2011")          # hifen nao separavel nos rotulos de metodo
IND = "\u2003"                                    # recuo dos metodos sob o problema
def sep(widths, aligns):
    """Linha separadora com larguras relativas (pandoc usa o numero de tracos)."""
    out = []
    for w, al in zip(widths, aligns):
        d = "-" * max(w, 3)
        out.append({"l": ":" + d, "r": d + ":", "c": ":" + d + ":"}[al])
    return "|" + "|".join(out) + "|"

# ---- campo -> texto, para o token generico {{v:prob:metodo:campo}} ----
FMT = {"cov": pct, "up": pct, "upm": pct, "w": e, "wmed": e, "bias": sgn, "IS": e,
       "short_mean": shf, "short_max": shf, "nmiss": str, "nmat": str, "lowmiss": str,
       "miss_pts_max": lambda x: f"{x:.0f}"}
def vtok(m):
    p, meth, f = m.group(1), m.group(2), m.group(3)
    return FMT[f](v[p][meth][f])

# ---- Tabelas 3 e 4: cobertura e largura ----
COLS = "| Cov. | Upper cov. | Mat. upper cov. | Mean width | Median width | Bias | IS |"
def cells(d):
    return (f"{pct(d['cov'])} | {pct(d['up'])} | {pct(d['upm'])} | {e(d['w'])} | {e(d['wmed'])} | "
            f"{sgn(d['bias'])} | {e(d['IS'])} |")
t1 = ["| Limit state / method " + COLS, sep([20, 9, 9, 10, 13, 13, 13, 13], "lrrrrrrr")]
for p in P:
    t1.append(f"| **{NAME[p]}** (P_f* = {e(pool[p], 2)}) | | | | | | | |")
    for m, s in MAIN: t1.append(f"| {IND}{NB(m)}{s} | " + cells(v[p][m]))
TABLE1 = "\n".join(t1)
t2 = ["| Method " + COLS, sep([15, 8, 10, 10, 14, 14, 15, 14], "lrrrrrrr")]
for m, s in MAIN: t2.append(f"| {NB(m)}{s} | " + cells(v["truss"][m]))
TABLE2 = "\n".join(t2)

# ---- Tabela 5: tamanho das falhas do limite superior ----
t3 = ["| Problem | Method | Misses | Material misses | Mean shortfall | Max shortfall | Largest miss (population points) |",
      sep([19, 15, 9, 11, 12, 12, 18], "llrrrrr")]
for p in P + ["truss"]:
    first = True
    for m in ("AK-MCS", "BSVR-perc", "Conformal"):
        d = v[p][m]
        if not d["nmiss"]: continue
        t3.append(f"| {NAME[p] if first else ''} | {NB(m)} | {d['nmiss']} | {d['nmat']} | {shf(d['short_mean'])} | "
                  f"{shf(d['short_max'])} | {d['miss_pts_max']:.0f} |"); first = False
TABLE3 = "\n".join(t3)

# ---- Tabela 6: grades de hiperparametros ----
nar, hp = N["narrow"], N["hiper"]
def cwb(d): return f"{pct(d['cov'])} / {e(d['w'])} / {sgn(d['bias'])}"
t4 = ["| Problem | Coarse grid (R=100) | Intermediate grid (R=50) | Extended grid (R=100) | Conformal, coarse | Conformal, extended |",
      sep([14, 25, 25, 25, 19, 19], "llllll")]
for p in ("d10", "truss"):
    t4.append(f"| {NAME[p]} | {cwb(nar[p]['methods']['BSVR-perc'])} | {cwb(hp[p]['methods']['BSVRwide-perc'])} | "
              f"{cwb(v[p]['BSVR-perc'])} | {pct(nar[p]['methods']['Conformal']['cov'])} / {e(nar[p]['methods']['Conformal']['w'])} | "
              f"{pct(v[p]['Conformal']['cov'])} / {e(v[p]['Conformal']['w'])} |")
TABLE4 = "\n".join(t4)

# ---- Tabela 7: controle do teto de C ----
cc = N["ctrlC"]; CCP = ("truss", "4b", "d10")
CCN = {"truss": "Truss (FE)", "4b": "Four-branch", "d10": "Linear, d=10"}
tc = ["| Problem | Percentile, C ≤ 10⁵ | Percentile, C ≤ 10⁷ | Gained / lost | BCa (mid‑rank) | Conformal | Fits with C > 10⁵ |",
      sep([13, 16, 16, 10, 17, 18, 10], "lllcllc")]
for p in CCP:
    d = cc[p]; b, w = d["base"], d["wide"]
    tc.append(f"| {CCN[p]} | {pct(b['BSVR-perc']['cov'])}\u00a0/\u00a0{pct(b['BSVR-perc']['upm'])} | "
              f"{pct(w['BSVR-perc']['cov'])}\u00a0/\u00a0{pct(w['BSVR-perc']['upm'])} | "
              f"{d['pair']['BSVR-perc']['gained']} / {d['pair']['BSVR-perc']['lost']} | "
              f"{pct(b['BSVR-BCa']['cov'])}\u00a0→\u00a0{pct(w['BSVR-BCa']['cov'])} | {pct(b['Conformal']['cov'])}\u00a0→\u00a0{pct(w['Conformal']['cov'])} | "
              f"{pr(w['fits']['big'], 0)} |")
TABLE_CTRL = "\n".join(tc)
# ---- Tabela 8: BCa, empates estritos vs meia-contagem ----
t5 = ["| Problem | Percentile | BCa, strict ties | BCa, mid-rank ties | Repetitions rescued / broken by mid-rank |",
      sep([19, 18, 18, 19, 21], "llllc")]
for p in P + ["truss"]:
    d = v[p]; q = N["v2_pair"][p]
    t5.append(f"| {NAME[p]} | {pct(d['BSVR-perc']['cov'])} / {pct(d['BSVR-perc']['up'])} | "
              f"{pct(d['BCa-strict']['cov'])} / {pct(d['BCa-strict']['up'])} | {pct(d['BSVR-BCa']['cov'])} / {pct(d['BSVR-BCa']['up'])} | "
              f"{q['rescued']} / {q['broken']} |")
TABLE5 = "\n".join(t5)

# ---- Tabela 1: problemas ----
pb = ["| Limit state | Definition | d | Exact P_f | P_f* (pop.) | SE_MC / P_f* | Regime |", sep([12, 23, 4, 15, 15, 11, 20], "llccccl")]
DEF = {"lin2": ("g=β−(1/√d)Σuᵢ, β=2.33", 2, "smooth, common"), "lin2r": ("β=3.09", 2, "smooth, rare"),
       "d10": ("β=2.5", 10, "smooth, high-dimensional"), "sph": ("g=r−‖u‖, r=3", 2, "curved, smooth"),
       "4b": ("series system, k=6", 2, "four failure branches")}
for p in P:
    ex = N["exact"][p]
    pb.append(f"| {NAME[p].split(',')[0]} | {DEF[p][0]} | {DEF[p][1]} | {e(ex,2) if ex else '—'} | {e(pool[p],2)} | "
              f"{pr(se[p]/pool[p],0)} | {DEF[p][2]} |")
TABLE_PROBLEMS = "\n".join(pb)


CAPTIONS = [
 ("Table 1. Analytical limit states", TABLE_PROBLEMS),
 ("Table 3. Coverage and sharpness on the analytical battery (N_call = 50, R = 100, nominal 95%)", TABLE1),
 ("Table 4. Coverage and sharpness on the truss fatigue problem (N_call = 50, R = 100, nominal 95%)", TABLE2),
 ("Table 5. Misses of the upper bound in 100 repetitions", TABLE3),
 ("Table 6. Percentile bootstrap (coverage / mean width / mean bias) under three hyperparameter grids, and the conformal bracket (coverage / mean width)", TABLE4),
 ("Table 7. Control on the upper bound of C (R = 50 per arm): coverage / material upper coverage", TABLE_CTRL),
 ("Table 8. Strict versus mid-rank treatment of ties in the BCa bias correction (coverage / upper coverage)", TABLE5),
]
out = "\n\n".join(f"**{c}**\n\n{t}" for c, t in CAPTIONS) + "\n"
out = re.sub(r"(P_f|P̂_f|Pr)\*(?!\*)", "\\1\u2217", out)
open(H/"tables.md", "w", encoding="utf-8").write(out); print("tables.md written:", len(CAPTIONS), "tables")
