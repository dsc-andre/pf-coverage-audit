"""RODADA v0.4 — reproduz TODAS as tabelas do artigo revisado.

Mudancas em relacao a run_final_wide.py:
  - AK-MCS corrigido (sem re-selecao de pontos ja avaliados);
  - referencia MC-CP (Monte Carlo simples + Clopper-Pearson, mesmo orcamento);
  - metricas de magnitude (cob.sup.mat, falta.rel) e log por repeticao em JSONL.
Grade do SVR: a ampliada (hiper.py). Sementes, pool, orcamento e R iguais aos
de run_final_wide.py.

Uso:
    python run_v2.py              # bateria analitica + trelica, R=100 (uma noite)
    python run_v2.py --so-trelica
    python run_v2.py --estreita   # experimento de mau ajuste (grade original)
    python run_v2.py --smoke
"""
import argparse, io, pathlib, sys, time
import numpy as np
import nivel1_benchmark as m
import fem_truss as ft
import hiper

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"; RESULTS.mkdir(exist_ok=True)


class Tee:
    def __init__(self, real): self.real, self.buf = real, io.StringIO()
    def write(self, s): self.real.write(s); self.real.flush(); self.buf.write(s)
    def flush(self): self.real.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--so-trelica", action="store_true")
    ap.add_argument("--estreita", action="store_true")
    a = ap.parse_args()
    m.N_POOL, m.N_CAND, m.N_INIT = 40_000, 5_000, 12
    m.N_BUDGET, m.CONF_NCAL = 50, 20
    m.B_BOOT, m.B_AL, m.M_POST, m.AK_SIM_K = 200, 20, 200, 2000
    m.R_REPS = 100
    if a.estreita:
        m.tune_svr = m.tune_svr_narrow
        m.METHODS = (m.bootstrap_svr, m.conformal_svr)
    if a.smoke:
        m.R_REPS, m.N_POOL, m.N_CAND = 5, 4000, 1000
        m.B_BOOT, m.B_AL, m.M_POST, m.AK_SIM_K = 40, 10, 60, 500
    nome = ("v2_estreita" if a.estreita else "v2") + ("_trelica" if a.so_trelica else "") \
           + ("_SMOKE" if a.smoke else "")
    m.REP_LOG = str(RESULTS / f"{nome}_reps.jsonl")
    pathlib.Path(m.REP_LOG).unlink(missing_ok=True)
    hiper.reset_counts(); t0 = time.time()
    tee = Tee(sys.stdout); real, sys.stdout = sys.stdout, tee
    try:
        if not a.so_trelica:
            m.RNG = np.random.default_rng(2025)
            for p in (m.Linear(2.33, 2), m.Linear(3.09, 2), m.Linear(2.5, 10),
                      m.Sphere(3.0, 2), m.FourBranch(6.0)):
                m.run(p)
        m.RNG = np.random.default_rng(2025)
        m.run(ft.TrussFatigue(m.RNG))
        print(); print(hiper.border_report())
    finally:
        sys.stdout = real
    (RESULTS / f"{nome}.txt").write_text(tee.buf.getvalue(), encoding="utf-8")
    print(f"\n>>> concluido em {(time.time()-t0)/60:.1f} min — results/{nome}.txt")


if __name__ == "__main__":
    main()
