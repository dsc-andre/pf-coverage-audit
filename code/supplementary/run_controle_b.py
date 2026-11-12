"""EXPERIMENTO DE CONTROLE — orcamento de treino equalizado.

Motivacao (critica de revisor): no estudo principal, o Conformal-SVR treina
com apenas n_train=30 pontos (os outros 20 do orcamento vao para calibracao),
enquanto Bootstrap-SVR e AK-MCS treinam com os 50. Parte da largura maior do
intervalo conformal pode, portanto, ser efeito do substituto mais fraco, e nao
do metodo — um confundimento na comparacao de NITIDEZ.

Este script isola isso: roda o Bootstrap-SVR com N_BUDGET=30 (mesmo numero de
pontos de TREINO do conformal) e o compara com o Conformal-SVR (30 treino + 20
calibracao). A diferenca de largura que sobrar e atribuivel ao METODO.

Como ler o resultado:
  - Se BSVR-30 continuar bem mais nitido E mantiver cobertura nos problemas
    faceis -> a largura do conformal e mesmo custo do metodo (nossa narrativa
    de nitidez se sustenta, so precisa da ressalva de orcamento).
  - Se BSVR-30 perder cobertura ou alargar muito -> boa parte da diferenca era
    orcamento, e o texto precisa ser corrigido nesse ponto.

OBS: o Bootstrap-30 gasta 30 chamadas verdadeiras e o Conformal gasta 50
(30+20). A comparacao e justa em TREINO, nao em orcamento total — que e
exatamente o eixo que queremos isolar. Reporte os dois numeros no texto.

Saida: results/controle_treino30.txt
Tempo estimado: ~2 h para os 5 analiticos + ~25 min para a trelica (R=100).

Uso:
    python run_controle_b.py            # completo (R=100)
    python run_controle_b.py --smoke    # R=5, verificacao rapida
"""
import argparse, io, pathlib, sys, time

import numpy as np
import nivel1_benchmark as m
import fem_truss as ft

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)

TRAIN_EQ = 30      # pontos de treino do conformal (N_BUDGET - CONF_NCAL)


def bootstrap_svr_train30(problem, pool):
    """Bootstrap-SVR treinado com 30 pontos (em vez de 50)."""
    old = m.N_BUDGET
    m.N_BUDGET = TRAIN_EQ
    try:
        entries, _ = m.bootstrap_svr(problem, pool)
    finally:
        m.N_BUDGET = old
    # renomeia para deixar claro no relatorio
    ren = {"BSVR-perc": "BSVR30-perc", "BSVR-BCa": "BSVR30-BCa"}
    return [(ren.get(lab, lab), pf, lo, hi) for lab, pf, lo, hi in entries], TRAIN_EQ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    # ---- configuracao identica a do estudo principal ----
    m.N_POOL, m.N_CAND, m.N_INIT = 40_000, 5_000, 12
    m.N_BUDGET, m.CONF_NCAL = 50, 20
    m.B_BOOT, m.B_AL, m.M_POST, m.AK_SIM_K = 200, 20, 200, 2000
    m.R_REPS = 100

    if args.smoke:
        m.R_REPS, m.N_POOL, m.N_CAND = 5, 4000, 1000
        m.B_BOOT, m.B_AL = 40, 10
        print("*** MODO SMOKE (R=5) — nao usar nos resultados ***")

    suffix = "_SMOKE" if args.smoke else ""

    print("=" * 86)
    print("  CONTROLE — orcamento de TREINO equalizado em 30 pontos")
    print(f"  R_REPS={m.R_REPS}  N_POOL={m.N_POOL}  B_BOOT={m.B_BOOT}")
    print("  BSVR30-*  : Bootstrap-SVR com 30 pontos de treino (30 chamadas)")
    print("  Conformal : 30 treino + 20 calibracao (50 chamadas)")
    if m.R_REPS < 100:
        print("  *** ATENCAO: R_REPS < 100 — nao usar nas tabelas finais ***")
    print("=" * 86 + "\n")

    # so os dois metodos que interessam ao controle
    m.METHODS = (bootstrap_svr_train30, m.conformal_svr)

    # saida vai para a TELA e para o buffer ao mesmo tempo (progresso ao vivo)
    class Tee:
        def __init__(self, real):
            self.real, self.buf = real, io.StringIO()
        def write(self, s):
            self.real.write(s); self.real.flush(); self.buf.write(s)
        def flush(self):
            self.real.flush()

    t0 = time.time()
    tee = Tee(sys.stdout)
    real_stdout, sys.stdout = sys.stdout, tee
    try:
        m.RNG = np.random.default_rng(2025)
        for prob in (m.Linear(beta=2.33, d=2), m.Linear(beta=3.09, d=2),
                     m.Linear(beta=2.5, d=10), m.Sphere(r=3.0, d=2),
                     m.FourBranch(k=6.0)):
            m.run(prob)
        m.RNG = np.random.default_rng(2025)
        m.run(ft.TrussFatigue(m.RNG))
    finally:
        sys.stdout = real_stdout

    text = tee.buf.getvalue()
    out = RESULTS / f"controle_treino30{suffix}.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\n>>> concluido em {(time.time()-t0)/60:.1f} min — saida em "
          f"results/{out.name}")


if __name__ == "__main__":
    main()