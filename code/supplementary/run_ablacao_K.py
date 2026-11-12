"""ABLACAO SOBRE K — o intervalo do AK-MCS depende do truncamento?

Motivacao (objecao de revisor, a mais grave): nossa propagacao do AK-MCS
sorteia realizacoes condicionais correlacionadas apenas nos K pontos do pool
com sinal menos certo e CONGELA o sinal dos demais. Isso subestima a incerteza
POR CONSTRUCAO: qualquer ponto fora do conjunto K que pudesse trocar de sinal
fica fora da variabilidade. Logo, a baixa cobertura na trelica (37%) admite
duas explicacoes concorrentes:
    (a) a variancia a posteriori do Kriging e superconfiante; ou
    (b) o truncamento em K subdispersa a propagacao.
O estudo principal NAO separa as duas. Este script separa.

COMO LER O RESULTADO:
  - Se a cobertura SUBIR muito com K (ex.: 37% -> 80%+), o problema e o
    TRUNCAMENTO. Nesse caso a conclusao sobre a variancia a posteriori deve ser
    retirada, e o metodo do artigo precisa usar K maior.
  - Se a cobertura ficar ESTAVEL com K, a explicacao (a) se sustenta e a
    conclusao sobre a variancia a posteriori pode ser reafirmada — com este
    experimento como prova.

CUSTO (medido): o passo caro e a decomposicao de Cholesky, O(K^3).
    K=500  ~4,5 s/rep      K=2000  ~6,0 s/rep      K=6000  ~22 s/rep
Com os tres valores e R=100, cada problema leva ~55 min.
Padrao: trelica + Linear d=10  ->  ~2 h no total.

Uso:
    python run_ablacao_K.py            # completo (R=100)
    python run_ablacao_K.py --smoke    # R=5, ~3 min
"""
import argparse, io, pathlib, sys, time

import numpy as np
import nivel1_benchmark as m
import fem_truss as ft

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)

K_VALUES = (500, 2000, 6000)


def make_ak(K):
    """AK-MCS com K fixo, rotulado."""
    def fn(problem, pool):
        old = m.AK_SIM_K
        m.AK_SIM_K = K
        try:
            entries, calls = m.ak_mcs(problem, pool)
        finally:
            m.AK_SIM_K = old
        return [(f"AK-K{K}", pf, lo, hi) for _, pf, lo, hi in entries], calls
    fn.__name__ = f"ak_mcs_K{K}"
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
    m.B_BOOT, m.B_AL, m.M_POST = 200, 20, 200
    m.R_REPS = 100
    Ks = K_VALUES

    if args.smoke:
        m.R_REPS, m.N_POOL, m.N_CAND = 5, 4000, 1000
        m.B_AL, m.M_POST = 10, 60
        Ks = (200, 800)
        print("*** MODO SMOKE — nao usar nos resultados ***")

    suffix = "_SMOKE" if args.smoke else ""

    print("=" * 86)
    print("  ABLACAO SOBRE K — truncamento da simulacao condicional (AK-MCS)")
    print(f"  R_REPS={m.R_REPS}  N_POOL={m.N_POOL}  N_BUDGET={m.N_BUDGET}  K={Ks}")
    print("  Cobertura sobe com K  -> o problema e o TRUNCAMENTO")
    print("  Cobertura estavel     -> o problema e a VARIANCIA a posteriori")
    if m.R_REPS < 100:
        print("  *** ATENCAO: R_REPS < 100 — nao usar nas tabelas finais ***")
    print("=" * 86 + "\n")

    m.METHODS = tuple(make_ak(K) for K in Ks)

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

    out = RESULTS / f"ablacao_K{suffix}.txt"
    out.write_text(tee.buf.getvalue(), encoding="utf-8")
    print(f"\n>>> concluido em {(time.time()-t0)/60:.1f} min — results/{out.name}")


if __name__ == "__main__":
    main()
