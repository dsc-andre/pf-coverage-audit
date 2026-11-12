"""REPROCESSAMENTO FINAL — bateria analitica + trelica com a grade ampliada.

Substitui as rodadas que geraram as Tabelas 1 e 2 do artigo. A unica mudanca em
relacao ao protocolo original e o ajuste de hiperparametros do SVR, que passa a
usar a grade ampliada de `hiper.py` (ver justificativa la). Tudo o mais —
orcamento, pool, numero de repeticoes, metodos, metricas, sementes — permanece
identico, de modo que a comparacao com os resultados anteriores e limpa.

Ao final imprime o DIAGNOSTICO DE BORDA: a fracao de ajustes cujo otimo caiu na
extremidade da grade. Esse numero vai para o artigo; ele responde de antemao a
objecao "e se voces tivessem ampliado mais?".

CUSTO: a grade ampliada tem ~6x mais combinacoes que a original, mas roda com
n_jobs=-1 (todos os nucleos). Estimativa: 4-8 h para a bateria analitica e
1-2 h para a trelica, dependendo do numero de nucleos.

Uso:
    python run_final_wide.py                # tudo (analitico + trelica), R=100
    python run_final_wide.py --so-trelica   # apenas a trelica
    python run_final_wide.py --smoke        # R=5, poucos minutos
"""
import argparse, io, pathlib, sys, time

import numpy as np
import nivel1_benchmark as m
import fem_truss as ft
import hiper

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)


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
    ap.add_argument("--so-trelica", action="store_true")
    args = ap.parse_args()

    # --- protocolo identico ao original, exceto o ajustador ---
    m.N_POOL, m.N_CAND, m.N_INIT = 40_000, 5_000, 12
    m.N_BUDGET, m.CONF_NCAL = 50, 20
    m.B_BOOT, m.B_AL, m.M_POST, m.AK_SIM_K = 200, 20, 200, 2000
    m.R_REPS = 100
    m.METHODS = (m.ak_mcs, m.bootstrap_svr, m.conformal_svr)

    m.tune_svr = hiper.tune_svr          # <<< a unica mudanca de metodo
    hiper.reset_counts()

    if args.smoke:
        m.R_REPS, m.N_POOL, m.N_CAND = 5, 4000, 1000
        m.B_BOOT, m.B_AL, m.M_POST, m.AK_SIM_K = 40, 10, 60, 500
        print("*** MODO SMOKE — nao usar nos resultados ***")

    suffix = "_SMOKE" if args.smoke else ""

    print("=" * 86)
    print("  REPROCESSAMENTO FINAL — grade de hiperparametros ampliada")
    print(f"  R_REPS={m.R_REPS}  N_POOL={m.N_POOL}  N_BUDGET={m.N_BUDGET}  B_BOOT={m.B_BOOT}")
    print(f"  grade: C={hiper.GRID['C']}")
    print(f"         gamma={hiper.GRID['gamma']}")
    print(f"         epsilon={hiper.GRID['epsilon']}")
    if m.R_REPS < 100:
        print("  *** ATENCAO: R_REPS < 100 — nao usar nas tabelas finais ***")
    print("=" * 86 + "\n")

    t0 = time.time()
    tee = Tee(sys.stdout)
    real, sys.stdout = sys.stdout, tee
    try:
        if not args.so_trelica:
            m.RNG = np.random.default_rng(2025)
            for prob in (m.Linear(beta=2.33, d=2), m.Linear(beta=3.09, d=2),
                         m.Linear(beta=2.5, d=10), m.Sphere(r=3.0, d=2),
                         m.FourBranch(k=6.0)):
                m.run(prob)
        m.RNG = np.random.default_rng(2025)
        m.run(ft.TrussFatigue(m.RNG))
        print()
        print(hiper.border_report())
    finally:
        sys.stdout = real

    nome = "final_wide_trelica" if args.so_trelica else "final_wide"
    out = RESULTS / f"{nome}{suffix}.txt"
    out.write_text(tee.buf.getvalue(), encoding="utf-8")
    print(f"\n>>> concluido em {(time.time()-t0)/60:.1f} min — results/{out.name}")


if __name__ == "__main__":
    main()
