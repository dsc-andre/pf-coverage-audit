"""CONTROLE DE C — a grade ampliada do SVR ainda esta truncada em C?

MOTIVACAO
Na rodada principal (run_v2.py) o otimo de C caiu no teto da grade (1e5) em 29%
dos ajustes; em 8% do total, sem que gamma estivesse no piso (o regime
quase-linear, que explica os outros 21%). Um revisor pode perguntar se a
cobertura do bootstrap mudaria com C maior. Este script responde.

O QUE FAZ
Roda Bootstrap-SVR e Conformal-SVR em tres problemas (trelica, Four-branch e
Linear d=10) com DOIS bracos que so diferem na faixa de C:
    base : C em {1, ..., 1e5}            (a grade do artigo)
    wide : C em {1, ..., 1e5, 1e6, 1e7}
Os dois bracos usam a mesma populacao e partem do mesmo estado do gerador em
cada repeticao, de modo que a comparacao e pareada.

TETO DE ITERACOES (importante)
Com C >= 1e6 o libsvm pode levar minutos num unico ajuste do Four-branch. Por
isso TODOS os ajustes deste controle (nos dois bracos) tem max_iter = 200 000.
O braco "base" difere, portanto, da rodada principal apenas por esse teto; a
comparacao base x rodada principal mostra se o teto importa, e a comparacao
base x wide mostra se a faixa de C importa. O script conta quantas vezes o
modelo selecionado bateu no teto de iteracoes.

COMO LER O RESULTADO
  - Cobertura parecida nos dois bracos  -> a grade do artigo nao estava
    truncada de modo relevante; entra no artigo como controle.
  - Cobertura muda com C maior          -> as tabelas do bootstrap dependem
    da grade tambem no teto de C, e o texto precisa dizer isso com os numeros.

RETOMAVEL: sao 6 celulas (problema x braco). Cada celula concluida fica
gravada; rodar de novo pula as prontas. Se cair no meio, e so relancar.

Uso (na mesma pasta de run_v2.py):
    python run_controle_C.py            # R=50 por celula
    python run_controle_C.py --rapido   # R=30 por celula
    python run_controle_C.py --smoke    # R=2, so para testar o ambiente
    python run_controle_C.py --tabela   # so remonta o resumo com o que ja rodou
Resultados: pasta results_controle_C, ao lado deste script.
"""
import argparse, collections, io, json, pathlib, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

import nivel1_benchmark as m
import fem_truss as ft
import hiper

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results_controle_C"

CAP = 200_000                                   # teto de iteracoes do libsvm (os dois bracos)
C_BASE = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
ARMS = {"base": C_BASE, "wide": C_BASE + [1e6, 1e7]}
PROBLEMS = ("truss", "4b", "d10")
NAMES = {"truss": "Trelica (FE), d=5", "4b": "Four-branch, d=2", "d10": "Linear, d=10"}
SEED = 2025
BUDGET, NCAL = 50, 20                           # como na rodada principal
N_POOL = 40_000
NPOOL_CUR = N_POOL                              # tamanho do pool em uso (4 000 no modo smoke)


def make_problem(key):
    if key == "truss": return ft.TrussFatigue(m.RNG)
    if key == "4b":    return m.FourBranch(6.0)
    if key == "d10":   return m.Linear(beta=2.5, d=10)
    raise ValueError(key)


def make_tuner(Cs, counts):
    """Mesmo ajustador de hiper.py, com a lista de C dada e teto de iteracoes."""
    grid = dict(hiper.GRID, C=list(Cs)); cmax = max(Cs)
    def tune(Xs, y):
        gs = GridSearchCV(SVR(kernel="rbf", max_iter=CAP), grid, cv=3,
                          scoring="neg_mean_squared_error", n_jobs=-1).fit(Xs, y)
        bp = dict(gs.best_params_)
        counts["total"] += 1
        counts[f"C={bp['C']:.0e}"] += 1
        if bp["C"] == cmax:
            counts["C_teto"] += 1
            if bp["gamma"] == 1e-5: counts["C_teto_gamma_piso"] += 1
        it = getattr(gs.best_estimator_, "n_iter_", None)   # ausente em versoes antigas do scikit-learn
        if it is not None and int(np.max(it)) >= CAP: counts["bateu_teto_iter"] += 1
        bp["max_iter"] = CAP                     # os ajustes seguintes (bootstrap etc.) herdam o teto
        return bp
    return tune


def metrics(rows):
    """Metricas de um metodo a partir das linhas por repeticao."""
    t = rows[0]["target"]; n = len(rows); se = np.sqrt(t*(1-t)/NPOOL_CUR)
    lo = np.array([r["lo"] for r in rows]); hi = np.array([r["hi"] for r in rows])
    pf = np.array([r["pf"] for r in rows]); miss = hi < t
    return dict(R=n, cov=float(np.mean((lo <= t) & (t <= hi))), up=float(np.mean(hi >= t)),
                upm=float(np.mean(hi >= t - se)), w=float(np.mean(hi-lo)), wmed=float(np.median(hi-lo)),
                bias=float(np.mean(pf-t)), nmiss=int(miss.sum()), nmat=int((hi < t-se).sum()),
                short_mean=float(((t-hi[miss])/t).mean()) if miss.any() else 0.0,
                short_max=float(((t-hi[miss])/t).max()) if miss.any() else 0.0, target=float(t))


def run_cell(prob, arm, R, smoke, tag):
    reps = OUT / f"reps_{prob}_{arm}{tag}.jsonl"
    reps.unlink(missing_ok=True)                 # celula interrompida recomeca do zero
    m.N_POOL, m.N_CAND, m.N_INIT = N_POOL, 5_000, 12
    m.N_BUDGET, m.CONF_NCAL = BUDGET, NCAL
    m.B_BOOT, m.B_AL = 200, 20
    if smoke:
        m.N_POOL, m.N_CAND, m.B_BOOT, m.B_AL = 4_000, 1_000, 40, 10
    m.R_REPS = R
    m.METHODS = (m.bootstrap_svr, m.conformal_svr)
    counts = collections.Counter(); m.tune_svr = make_tuner(ARMS[arm], counts)
    m.REP_LOG = str(reps)
    m.RNG = np.random.default_rng(SEED)          # mesma populacao e mesmo fluxo nos dois bracos
    t0 = time.time()
    print(f"\n{'='*104}\nCELULA: {NAMES[prob]} | braco {arm} (C ate {max(ARMS[arm]):.0e}) | R={R}\n{'='*104}", flush=True)
    m.run(make_problem(prob))
    return dict(counts=dict(counts), minutos=round((time.time()-t0)/60, 1), R=R)


def load_rows(prob, arm, tag):
    f = OUT / f"reps_{prob}_{arm}{tag}.jsonl"
    by = collections.defaultdict(list)
    for line in f.open(encoding="utf-8"):
        r = json.loads(line); by[r["method"]].append(r)
    return by


def tabela(done, tag):
    L = ["", "=" * 104, "RESUMO — controle de C (base: C ate 1e5 | wide: C ate 1e7; teto de 200 000 iteracoes nos dois)", "=" * 104]
    for prob in PROBLEMS:
        if not all(f"{prob}_{a}" in done for a in ARMS): continue
        rows = {a: load_rows(prob, a, tag) for a in ARMS}
        t = rows["base"]["BSVR-perc"][0]["target"]
        L += ["", f"{NAMES[prob]}   Pf(pool) = {t:.4e}   R = {done[prob + '_base']['R']}",
              f"  {'metodo':<11}{'braco':<6}{'cob.':>6}{'cob.sup':>9}{'sup.mat':>9}{'falhas':>8}{'mater.':>8}"
              f"{'falta med.':>12}{'falta max':>11}{'larg.media':>12}{'larg.med.':>11}{'vies':>12}"]
        for meth in ("BSVR-perc", "BSVR-BCa", "Conformal"):
            for a in ARMS:
                d = metrics(rows[a][meth])
                L.append(f"  {meth:<11}{a:<6}{d['cov']:>6.0%}{d['up']:>9.0%}{d['upm']:>9.0%}{d['nmiss']:>8d}{d['nmat']:>8d}"
                         f"{d['short_mean']:>12.1%}{d['short_max']:>11.1%}{d['w']:>12.2e}{d['wmed']:>11.2e}{d['bias']:>+12.2e}")
            b, w = rows["base"][meth], rows["wide"][meth]
            inn = lambda r: r["lo"] <= r["target"] <= r["hi"]
            n = min(len(b), len(w))
            ganhou = sum((not inn(b[i])) and inn(w[i]) for i in range(n)); perdeu = sum(inn(b[i]) and (not inn(w[i])) for i in range(n))
            L.append(f"    pareado ({meth}): wide cobre e base nao em {ganhou} repeticoes; base cobre e wide nao em {perdeu}")
        for a in ARMS:
            c = done[f"{prob}_{a}"]["counts"]; n = max(c.get("total", 0), 1)
            cs = ", ".join(f"{k[2:]}: {100*v/n:.0f}%" for k, v in sorted(((k, v) for k, v in c.items() if k.startswith("C=")),
                                                                         key=lambda kv: float(kv[0][2:])))
            L.append(f"  [{a}] {n} ajustes | C escolhido -> {cs}")
            L.append(f"  [{a}] C no teto da grade: {100*c.get('C_teto', 0)/n:.1f}% (com gamma no piso: {100*c.get('C_teto_gamma_piso', 0)/n:.1f}%)"
                     f" | modelo selecionado bateu no teto de iteracoes: {100*c.get('bateu_teto_iter', 0)/n:.1f}%"
                     f" | {done[prob + '_' + a]['minutos']} min")
    txt = "\n".join(L)
    print(txt)
    (OUT / f"controle_C{tag}.txt").write_text(txt + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--tabela", action="store_true")
    a = ap.parse_args()
    global NPOOL_CUR
    OUT.mkdir(exist_ok=True)
    NPOOL_CUR = 4_000 if a.smoke else N_POOL
    R = 2 if a.smoke else (30 if a.rapido else 50)
    tag = "_SMOKE" if a.smoke else ""
    prog = OUT / f"progresso{tag}.json"
    done = json.loads(prog.read_text(encoding="utf-8")) if prog.exists() else {}
    if a.smoke: print("*** MODO SMOKE — so para testar o ambiente; nao usar os numeros ***")
    if not a.tabela:
        t0 = time.time()
        for prob in PROBLEMS:
            for arm in ARMS:
                key = f"{prob}_{arm}"
                if key in done and done[key]["R"] == R:
                    print(f"celula {key}: ja concluida ({done[key]['minutos']} min) — pulando"); continue
                done[key] = run_cell(prob, arm, R, a.smoke, tag)
                prog.write_text(json.dumps(done, indent=1), encoding="utf-8")
                print(f">>> celula {key} concluida em {done[key]['minutos']} min", flush=True)
        print(f"\n>>> todas as celulas concluidas ({(time.time()-t0)/60:.1f} min nesta execucao)")
    tabela(done, tag)
    print(f"\n>>> arquivos em: {OUT}")
    print(f">>> mandar de volta: controle_C{tag}.txt, progresso{tag}.json e os seis reps_*{tag}.jsonl")


if __name__ == "__main__":
    main()
