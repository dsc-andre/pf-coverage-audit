"""Tokens escalares do texto, todos derivados de numbers.json (versao da rodada v2).

Os numeros das tabelas principais entram no texto pelo token generico
{{v:problema:metodo:campo}} resolvido em fill.py; aqui ficam so os derivados."""
import json, pathlib
from scipy.stats import norm as _n
H = pathlib.Path(__file__).resolve().parent; N = json.load(open(H/"numbers.json"))
v, pool, se = N["v2"], N["v2_pool"], N["v2_se"]
def _sup(n): return str(n).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
def e(x, d=1): return "0" if abs(x) < 1e-12 else (lambda m, ex: f"{m}×10{_sup(int(ex))}")(*f"{x:.{d}e}".split("e"))
def sg(x, d=1): return ("+" if x > 0 else "−" if x < 0 else "") + e(abs(x), d)
def pc(x): return f"{x:.0f}%"
def pr(x, d=1): return f"{100*x:.{d}f}%"
P = ["lin2", "lin2r", "d10", "sph", "4b", "truss"]; LIN = ("lin2", "lin2r", "d10")
T = {}

# ---- alvo e erro de Monte Carlo ----
T["pool_truss"] = e(pool["truss"], 3); T["se_truss"] = e(se["truss"], 1)
T["se_truss_rel"] = pr(se["truss"]/pool["truss"]); T["se_4b_rel"] = pr(se["4b"]/pool["4b"])
q = [100*se[p]/pool[p] for p in P]; T["se_rel_range"] = f"{min(q):.0f}% and {max(q):.0f}%"
g = N["gov"]; T["gov0"] = pc(100*g["bottom"]/g["total"]); T["gov2"] = pc(100*g["top"]/g["total"])

# ---- 5.1 ----

r = [v[p]["MC-CP"]["w"]/max(v[p]["BSVR-perc"]["w"], 1e-30) for p in LIN]
assert 50 < min(r) and max(r) < 5e4, r
T["mc_zero_range"] = f"{100*min(N['mc_zero'].values()):.0f}% to {100*max(N['mc_zero'].values()):.0f}%"
T["mc_k0_hi"] = f"{1-0.025**(1/50):.3f}"                       # Clopper-Pearson com k=0, n=50
c = N["collapse"]["d10"]; assert len(c) == 1; c = c[0]
T["d10_col_pf"] = e(c["pf"]); T["d10_col_lo"] = e(c["lo"], 0); T["d10_col_hi"] = e(c["hi"]); T["d10_col_t"] = e(c["target"])
T["d10_col_short"] = pr((c["target"]-c["hi"])/c["target"], 0)
sph = [v["sph"][m]["cov"] for m in ("AK-MCS", "BSVR-perc", "BSVR-BCa", "Conformal")]
T["sph_range"] = f"{min(sph)}–{max(sph)}%"
b4 = v["4b"]["BSVR-perc"]
T["B_4b_biasrel"] = f"−{abs(100*b4['bias']/pool['4b']):.0f}%"; T["B_4b_biasmed"] = f"{abs(100*b4['bias_med']/pool['4b']):.0f}%"
T["conf4b_frac"] = {0.25: "a quarter"}.get(round(N["conf4b_frac_wider_mc"], 2), pr(N["conf4b_frac_wider_mc"], 0))
T["n4b_col"] = str(len(N["collapse"]["4b"]))

# ---- 5.2 ----
a = v["truss"]["AK-MCS"]; T["AK_tr_wpts"] = f"{a['w']*40000:.0f}"
k = N["ablK_fix"]; T["abl_K"] = (", ".join(pc(k[str(K)]["cov"]) for K in (500, 2000, 6000)) +
     f" (upper coverage {min(k[str(K)]['up'] for K in (500,2000,6000))}–{max(k[str(K)]['up'] for K in (500,2000,6000))}%)")
kn = N["kern"]; rep = N["ak"]["truss"]          # replicacao independente do kernel principal, mesma populacao
T["ISO_rep_cov"] = pc(rep["cov"]); T["ISO_rep_up"] = pc(rep["up"])
T["ARDRBF_cov"] = pc(kn["ardrbf"]["cov"]); T["ARDRBF_up"] = pc(kn["ardrbf"]["up"])
T["ISOMAT_cov"] = pc(kn["isomatern"]["cov"]); T["ISOMAT_up"] = pc(kn["isomatern"]["up"])
T["ARD_cov"] = pc(N["akard"]["cov"]); T["ARD_up"] = pc(N["akard"]["up"])
x = v["truss"]["MC-CP"]["w"]/v["truss"]["Conformal"]["w"]
T["C_tr_vs_mc"] = {8: "eight"}.get(round(x), f"{x:.0f}")

# ---- 5.3 ----
T["AK_tr_maxpts"] = {1: "one", 2: "two"}[round(a["miss_pts_max"])]
bt = v["truss"]["BSVR-perc"]; T["B_tr_nimm"] = str(bt["nmiss"]-bt["nmat"])
c = N["collapse"]["truss"]; assert len(c) == 1; c = c[0]
T["tr_col_pf"] = e(c["pf"]); T["tr_col_hi"] = e(c["hi"]); T["tr_col_t"] = e(c["target"])
x = c["target"]/c["pf"]; T["tr_col_x"] = {25: "twenty-five"}.get(round(x), f"{x:.0f}")
T["tr_col_short"] = pr((c["target"]-c["hi"])/c["target"], 0)
nrep = 100*len(P)
T["n_reps_all"] = str(nrep)
T["C_nmat_all"] = str(sum(v[p]["Conformal"]["nmat"] for p in P)); assert T["C_nmat_all"] == "1"
T["AK_nmat_all"] = str(sum(v[p]["AK-MCS"]["nmat"] for p in P)); assert T["AK_nmat_all"] == "0"
ak = N["ak"]                                     # replicacao do AK-MCS em populacoes independentes (R=100 cada)
nm = {p: round(ak[p]["R"]*(1-ak[p]["upm"]/100)) for p in P}
T["AKrep_nmat"] = str(sum(nm.values())); T["AKrep_n"] = str(sum(ak[p]["R"] for p in P))
assert nm["sph"] > 0 and nm["4b"] > 0 and sum(nm.values()) == nm["sph"] + nm["4b"]
T["AKrep_sph"] = f"{nm['sph']}"; T["AKrep_4b"] = f"{nm['4b']}"
T["AKrep_sph_max"] = pr(ak["sph"]["short_max"], 0); T["AKrep_4b_max"] = pr(ak["4b"]["short_max"], 0)
T["AKrep_4b_cov"] = pc(ak["4b"]["cov"])
T["AK_mat_rate"] = pr((int(T["AK_nmat_all"]) + sum(nm.values()))/(nrep + sum(ak[p]["R"] for p in P)), 1)
T["AK_mat_n"] = str(int(T["AK_nmat_all"]) + sum(nm.values())); T["AK_mat_tot"] = f"{nrep + sum(ak[p]['R'] for p in P):,}"
# replicacao do bootstrap e do conformal (final_wide.txt, populacoes diferentes)
fw = N["fw"]
dB = max(abs(fw[p]["methods"]["BSVR-perc"]["cov"]-v[p]["BSVR-perc"]["cov"]) for p in P)
dC = max(abs(fw[p]["methods"]["Conformal"]["cov"]-v[p]["Conformal"]["cov"]) for p in P)
assert dB <= 6 and dC <= 6, (dB, dC)
T["rep_maxdiff"] = str(max(dB, dC))
T["rep_B_4b"] = pc(fw["4b"]["methods"]["BSVR-perc"]["cov"]); T["rep_B_tr"] = pc(fw["truss"]["methods"]["BSVR-perc"]["cov"])
T["rep_C_tr"] = pc(fw["truss"]["methods"]["Conformal"]["cov"])

# ---- 5.4 ----
nar, hp = N["narrow"], N["hiper"]
T["grid_covs"] = (f"{pc(nar['d10']['methods']['BSVR-perc']['cov'])}, {pc(hp['d10']['methods']['BSVRwide-perc']['cov'])} or "
                  f"{pc(v['d10']['BSVR-perc']['cov'])}")
T["grid_lo"] = pc(nar["d10"]["methods"]["BSVR-perc"]["cov"]); T["grid_hi"] = pc(v["d10"]["BSVR-perc"]["cov"])
T["narrow_tr_bias"] = sg(nar["truss"]["methods"]["BSVR-perc"]["bias"])
T["narrow_C_w"] = f"{e(nar['d10']['methods']['Conformal']['w'])} in ten dimensions, {e(nar['truss']['methods']['Conformal']['w'])} on the truss"
T["narrow_C_tr_cov"] = pc(nar["truss"]["methods"]["Conformal"]["cov"]); T["narrow_C_d10_cov"] = pc(nar["d10"]["methods"]["Conformal"]["cov"])
T["narrow_C_d10_rep"] = pc(N["treino30"]["d10"]["methods"]["Conformal"]["cov"])
T["narrow_C_tr_rep"] = pc(N["treino30"]["truss"]["methods"]["Conformal"]["cov"])
_r = min(nar[p]["methods"]["Conformal"]["w"]/v[p]["MC-CP"]["w"] for p in ("d10", "truss"))
assert _r > 2; T["narrow_C_vs_mc"] = "more than twice"
b = N["border"]; tot = b["total"]
T["edge_n"] = f"{tot:,}"; T["edge_eps_min"] = pr(b["epsilon"]["min"]/tot, 0)
T["edge_C_max"] = pr(b["C"]["max"]/tot, 0); T["edge_ridge"] = pr(b["ridge"]/tot, 0)
T["edge_ridge_share"] = pr(b["ridge"]/b["C"]["max"], 0); T["edge_C_alone"] = pr((b["C"]["max"]-b["ridge"])/tot, 0)

# ---- 5.5 (curva de orcamento: rodadas separadas, R=30) ----
cv = N["curve"]
def seq(prob, m): return ", ".join(f"{cv[prob][str(n)][m]['cov']:.0f}%" for n in (30, 50, 80, 120, 150))
T["curve_B_tr"] = seq("truss", "BSVR-perc") + " at 30 to 150 calls"
T["curve_C_tr"] = ", ".join(f"{cv['truss'][str(n)]['Conformal']['cov']:.0f}%" for n in (50, 80, 120, 150)) + " at 50 to 150 calls"
T["curve_BCa_tr"] = seq("truss", "BSVR-BCa")
x = [cv["d10"][str(n)]["BSVR-BCa"]["cov"] for n in (30, 50, 80, 120, 150)]; T["curve_BCa_d10"] = f"{min(x):.0f}–{max(x):.0f}%"
_b = cv["truss"]
T["curve_B_w50"] = e(_b["50"]["BSVR-perc"]["w"]); T["curve_B_w150"] = e(_b["150"]["BSVR-perc"]["w"])
_bb = [_b[str(n)]["BSVR-perc"]["bias"] for n in (80, 120, 150)]
T["curve_B_bias"] = f"{sg(max(_bb, key=abs), 0)} to {sg(min(_bb, key=abs), 0)}"
T["curve_B_biasrel"] = f"{100*abs(sum(_bb)/3)/pool['truss']:.0f}%"
T["curve_B_bias30"] = sg(_b["30"]["BSVR-perc"]["bias"], 0)
T["curve_C_low"] = f"{cv['d10']['30']['Conformal']['cov']:.0f}% in ten dimensions and {cv['truss']['30']['Conformal']['cov']:.0f}% on the truss at 30 calls"

# ---- 5.6 (BCa e empates) ----
def _up(tie): return f"{_n.cdf(2*_n.ppf((1-tie)/2) + _n.ppf(0.975)):.2f}"
T["tie40_p0"] = f"{(1-0.4)/2:.2f}"; T["tie40_up"] = _up(0.4); T["tie50_up"] = _up(0.5)
s_ = [v[p]["BCa-strict"]["cov"] for p in LIN]; m_ = [v[p]["BSVR-BCa"]["cov"] for p in LIN]; p_ = [v[p]["BSVR-perc"]["cov"] for p in LIN]
T["BCa_strict_lin"] = f"{min(s_)}–{max(s_)}%"; T["BCa_mid_lin"] = f"{min(m_)}–{max(m_)}%"; T["perc_lin"] = f"{min(p_)}–{max(p_)}%"
d_ = [m - s for m, s in zip(m_, s_)]; T["BCa_gain_lin"] = f"{min(d_)} to {max(d_)}"
pp = N["v2_pair"]; rs = [pp[p]["rescued"] for p in LIN]; br = [pp[p]["broken"] for p in LIN]
T["BCa_resc_lin"] = f"{min(rs)} to {max(rs)}"; T["BCa_brk_lin"] = str(max(br))
assert pp["sph"]["rescued"] == 0 and pp["sph"]["broken"] == 0
assert all(v[p]["BSVR-BCa"]["cov"] <= v[p]["BSVR-perc"]["cov"] for p in P)      # BCa nunca supera o percentil
T["BCa_lin2_low"] = str(v["lin2"]["BSVR-BCa"]["lowmiss"])

# ---- 6.x ----
x = v["sph"]["Conformal"]["w"]/v["sph"]["BSVR-perc"]["w"]; T["C_sph_ratio"] = {5: "five"}.get(round(x), f"{x:.0f}")
x = v["truss"]["Conformal"]["w"]/pool["truss"]; assert 1.2 < x < 1.6; T["C_tr_wrel"] = f"{x:.1f}"
T["C_min_cov"] = pc(min(v[p]["Conformal"]["cov"] for p in P))
json.dump(T, open(H/"tokens.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(T, indent=1, ensure_ascii=False))
