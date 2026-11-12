"""Figuras do artigo revisado (v2) — leem numbers.json (fonte unica)."""
import json, pathlib, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE/"figures"; OUT.mkdir(exist_ok=True)
N = json.load(open(HERE/"numbers.json"))
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linewidth": 0.5, "figure.dpi": 150, "savefig.bbox": "tight", "axes.axisbelow": True})
P = ["lin2", "lin2r", "d10", "sph", "4b", "truss"]
LAB = ["Linear\nd=2 (1e-2)", "Linear\nd=2 (1e-3)", "Linear\nd=10", "Sphere\nd=2", "Four-branch\nd=2", "Truss (FE)\nd=5"]
M = ["AK-MCS", "BSVR-perc", "BSVR-BCa", "Conformal"]
col = {"AK-MCS": "#6E6E6E", "BSVR-perc": "#2C6FBB", "BSVR-BCa": "#7EB0E0", "Conformal": "#D9822B", "MC-CP": "#3B8B5A"}
def get(p, m):
    return N["v2"][p][m]                 # rodada v2: todos os metodos no mesmo pool, R=100
# ---------------- FIG 2: cobertura + largura relativa ao MC-CP ----------------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True, gridspec_kw={"hspace": 0.16})
fig.subplots_adjust(top=0.91)
x = np.arange(len(P)); w = 0.2
a1.axhspan(91, 99, color="#9ecae1", alpha=0.18, zorder=0); a1.axhline(95, color="#333", ls="--", lw=1)
for k, m in enumerate(M):
    v = [get(p, m)["cov"] for p in P]
    a1.bar(x+(k-1.5)*w, v, w, color=col[m], label=m, edgecolor="white", lw=0.4, zorder=2)
a1.set_ylim(0, 108); a1.set_ylabel("coverage (%)")
fig.legend(*a1.get_legend_handles_labels(), ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.995), fontsize=8.5)
a1.set_title("(a) empirical coverage of nominal 95% intervals", fontsize=9.5, loc="left")
for k, m in enumerate(M):
    v = [max(get(p, m)["w"], 1e-7)/N["v2"][p]["MC-CP"]["w"] for p in P]
    a2.bar(x+(k-1.5)*w, v, w, color=col[m], edgecolor="white", lw=0.4, zorder=2)
a2.axhline(1, color=col["MC-CP"], lw=1.4); a2.text(5.45, 1.25, "crude MC + Clopper–Pearson", color=col["MC-CP"], fontsize=7.5, ha="right")
a2.set_yscale("log"); a2.set_ylim(5e-7, 5)
a2.set_title("(b) mean width relative to the crude Monte Carlo reference (same 50 calls)", fontsize=9.5, loc="left")
a2.set_xticks(x); a2.set_xticklabels(LAB, fontsize=8)
fig.savefig(OUT/"fig2_coverage.png", dpi=200); fig.savefig(OUT/"fig2_coverage.pdf"); plt.close(fig)
# ---------------- FIG 3: curva de orcamento ----------------
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
mk = {"AK-MCS": "o", "BSVR-perc": "s", "BSVR-BCa": "^", "Conformal": "D"}
for ax, prob, ttl in zip(axs, ("d10", "truss"), ("Linear, d=10", "Truss (FE), d=5")):
    ax.axhspan(87, 100, color="#9ecae1", alpha=0.18, zorder=0); ax.axhline(95, color="#333", ls="--", lw=1)
    for m in M:
        xs, ys = [], []
        for Nb, d in sorted(N["curve"][prob].items(), key=lambda t: int(t[0])):
            if m in d: xs.append(int(Nb)); ys.append(d[m]["cov"])
        ax.plot(xs, ys, marker=mk[m], color=col[m], label=m, lw=1.4, ms=5)
    ax.set_title(ttl, fontsize=9.5); ax.set_xlabel("budget $N_{call}$"); ax.set_xticks([30, 50, 80, 120, 150])
axs[0].set_ylabel("coverage (%)"); axs[0].set_ylim(0, 105)
axs[1].legend(frameon=False, fontsize=8, loc="lower right")
fig.savefig(OUT/"fig3_budget.png", dpi=200); fig.savefig(OUT/"fig3_budget.pdf"); plt.close(fig)
print("figuras ok")
