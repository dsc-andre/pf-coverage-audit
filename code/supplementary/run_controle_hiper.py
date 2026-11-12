"""CONTROLE DE HIPERPARAMETROS — a falha do bootstrap e ma escolha de C/gamma?

Motivacao: o estudo principal ajusta o SVR por GridSearchCV com
    C in {1, 10, 100}, gamma in {"scale", 0.1, 1.0}, epsilon in {0.01, 0.1}
com cv=3 e criterio de erro quadratico medio (MSE). Tres fragilidades sao
plausiveis e um revisor pode levanta-las:
    (i)  a grade e grossa e TRUNCADA — se o C otimo estiver acima de 100, a
         busca escolhe a borda e ninguem percebe;
    (ii) o criterio e o MSE GLOBAL, mas o que importa em confiabilidade e
         acertar o SINAL perto do estado-limite;
    (iii) com 30-50 pontos e cv=3, cada fold treina com 20-33 pontos e a
         estimativa de erro e ruidosa, tornando a escolha quase aleatoria.

Se a sub-cobertura do bootstrap vier de hiperparametro mal escolhido, e nao de
ma especificacao do modelo, ampliar a grade e/ou mudar o criterio deve melhorar
a cobertura. Este script testa isso.

VARIANTES COMPARADAS (todas com bootstrap-SVR, mesmo orcamento):
    BSVR-base   grade atual (referencia — deve reproduzir a Tabela 1/2)
    BSVR-wide   grade ampliada: C ate 1e4, gamma em faixa log mais larga
    BSVR-sign   grade ampliada + criterio focado na regiao do estado-limite
                (MSE calculado apenas nos pontos de |g| menor, isto e, nos
                 que decidem o sinal)

COMO LER O RESULTADO:
  - Cobertura sobe muito com wide/sign -> a falha era de ajuste; o artigo
    precisa corrigir o protocolo e refazer as tabelas.
  - Cobertura permanece baixa -> confirma ma especificacao do modelo, e o
    artigo ganha um quarto controle. (E o esperado: intervalos ESTREITOS com
    vies GRANDE indicam replicas concordando entre si e erradas juntas, nao
    hiperparametro instavel, que produziria intervalos LARGOS.)

CUSTO: a grade ampliada tem ~3x mais combinacoes; o script usa n_jobs=-1 para
usar todos os nucleos. Estimativa: ~1,5 a 3 h para os dois problemas com R=50.
Rode primeiro com --smoke (poucos minutos) para calibrar a expectativa.

Uso:
    python run_controle_hiper.py            # R=50
    python run_controle_hiper.py --smoke    # R=5
"""
import argparse, io, pathlib, sys, time

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

import nivel1_benchmark as m
import fem_truss as ft

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)

GRID_BASE = {"C": [1, 10, 100],
             "gamma": ["scale", 0.1, 1.0],
             "epsilon": [0.01, 0.1]}

GRID_WIDE = {"C": [1, 10, 100, 1000, 10000],
             "gamma": ["scale", 0.01, 0.1, 0.5, 1.0, 5.0],
             "epsilon": [0.001, 0.01, 0.1]}


def scorer_near_limit(estimator, X, y):
    """MSE restrito aos pontos mais proximos do estado-limite (|g| menor).

    Sao esses que decidem o sinal e, portanto, a probabilidade de falha.
    Retorna score negativo (maior = melhor), como exige o sklearn.
    """
    pred = estimator.predict(X)
    if len(y) < 4:
        return -float(np.mean((y - pred) ** 2))
    thr = np.quantile(np.abs(y), 0.5)          # metade mais proxima de g=0
    msk = np.abs(y) <= thr
    if msk.sum() < 2:
        msk = np.ones(len(y), bool)
    return -float(np.mean((y[msk] - pred[msk]) ** 2))


def make_tuner(grid, scoring):
    def tune(Xs, y):
        return GridSearchCV(SVR(kernel="rbf"), grid, cv=3, scoring=scoring,
                            n_jobs=-1).fit(Xs, y).best_params_
    return tune


def make_variant(label, grid, scoring):
    """Roda bootstrap_svr com um ajustador de hiperparametros trocado."""
    def fn(problem, pool):
        orig = m.tune_svr
        m.tune_svr = make_tuner(grid, scoring)
        try:
            entries, calls = m.bootstrap_svr(problem, pool)
        finally:
            m.tune_svr = orig
        out = []
        for lab, pf, lo, hi in entries:
            tag = "perc" if "perc" in lab else "BCa"
            out.append((f"{label}-{tag}", pf, lo, hi))
        return out, calls
    fn.__name__ = f"bootstrap_{label}"
    return fn


class Tee:
    def __init__(self, real):
        self.real, self.buf = real, io.StringIO()
    def write(self, s):
        self.real.write(s); self.real.flush(); self.buf.write(s)
    def flush(self):
        self.real.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    m.N_POOL, m.N_CAND, m.N_INIT = 40_000, 5_000, 12
    m.N_BUDGET, m.CONF_NCAL = 50, 20
    m.B_BOOT, m.B_AL, m.M_POST, m.AK_SIM_K = 200, 20, 200, 2000
    m.R_REPS = 50

    if args.smoke:
        m.R_REPS, m.N_POOL, m.N_CAND = 5, 4000, 1000
        m.B_BOOT, m.B_AL = 40, 10
        print("*** MODO SMOKE — nao usar nos resultados ***")

    suffix = "_SMOKE" if args.smoke else ""

    print("=" * 86)
    print("  CONTROLE DE HIPERPARAMETROS DO SVR")
    print(f"  R_REPS={m.R_REPS}  N_POOL={m.N_POOL}  N_BUDGET={m.N_BUDGET}")
    print("  base = grade atual | wide = grade ampliada | sign = wide + criterio")
    print("         focado na regiao do estado-limite")
    print("  Cobertura sobe -> a falha era de ajuste")
    print("  Cobertura estavel -> confirma ma especificacao do modelo")
    print("=" * 86 + "\n")

    m.METHODS = (
        make_variant("BSVRbase", GRID_BASE, "neg_mean_squared_error"),
        make_variant("BSVRwide", GRID_WIDE, "neg_mean_squared_error"),
        make_variant("BSVRsign", GRID_WIDE, scorer_near_limit),
    )

    t0 = time.time()
    tee = Tee(sys.stdout)
    real, sys.stdout = sys.stdout, tee
    try:
        m.RNG = np.random.default_rng(2025)
        m.run(ft.TrussFatigue(m.RNG))
        m.RNG = np.random.default_rng(2025)
        m.run(m.Linear(beta=2.5, d=10))
    finally:
        sys.stdout = real

    out = RESULTS / f"controle_hiper{suffix}.txt"
    out.write_text(tee.buf.getvalue(), encoding="utf-8")
    print(f"\n>>> concluido em {(time.time()-t0)/60:.1f} min — results/{out.name}")


if __name__ == "__main__":
    main()
