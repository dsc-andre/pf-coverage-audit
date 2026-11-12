"""Reproduz todos os resultados do artigo com um comando so.

Executa (1) a bateria analitica e (2) o caso FEM da trelica, ambos com R=100,
salvando as saidas em ../results/. Depois gera as figuras em ../figures/.

    python run_all.py

AVISO: a execucao completa leva varias horas em CPU de desktop (R=100 em seis
problemas). Para um teste rapido de fumaca, use:

    python run_all.py --smoke     (R=5, pool reduzido; NAO usar nos resultados)
"""
import argparse, contextlib, io, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)


def tee(text, path):
    """Imprime na tela e grava em arquivo."""
    sys.stdout.write(text)
    sys.stdout.flush()
    path.write_text(text, encoding="utf-8")


def run_block(label, fn, outfile):
    print(f"\n>>> {label} — iniciando ({time.strftime('%H:%M:%S')})")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    text = buf.getvalue()
    tee(text, RESULTS / outfile)
    print(f">>> {label} — concluido, saida em results/{outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="rodada rapida de verificacao (R=5); nao usar nos resultados")
    args = ap.parse_args()

    import numpy as np
    import nivel1_benchmark as m
    import fem_truss as ft

    if args.smoke:
        m.R_REPS, m.N_POOL, m.N_CAND = 5, 4000, 1000
        m.B_BOOT, m.B_AL, m.M_POST, m.AK_SIM_K = 40, 10, 60, 500
        print("*** MODO SMOKE (R=5) — resultados NAO servem para o artigo ***")

    # sufixo separado no modo smoke: NAO sobrescrever os resultados de referencia
    suffix = "_SMOKE" if args.smoke else "_R100"

    # ---- 1. bateria analitica ----
    def analytical():
        m.RNG = np.random.default_rng(2025)
        print(f"CONFIG: R_REPS={m.R_REPS} N_POOL={m.N_POOL} N_BUDGET={m.N_BUDGET}\n")
        for prob in (m.Linear(beta=2.33, d=2), m.Linear(beta=3.09, d=2),
                     m.Linear(beta=2.5, d=10), m.Sphere(r=3.0, d=2),
                     m.FourBranch(k=6.0)):
            m.run(prob)

    run_block("Bateria analitica", analytical, f"analytical{suffix}.txt")

    # ---- 2. caso FEM ----
    def fem():
        m.RNG = np.random.default_rng(2025)
        if not args.smoke:
            m.N_POOL, m.B_BOOT, m.B_AL = 40_000, 200, 20
            m.M_POST, m.AK_SIM_K, m.N_CAND = 200, 2000, 5_000
        truss = ft.TrussFatigue(m.RNG)
        print(f"CONFIG: R_REPS={m.R_REPS} N_POOL={m.N_POOL}\n")
        m.run(truss)

    run_block("Caso FEM (trelica)", fem, f"truss{suffix}.txt")

    # ---- 3. figuras ----
    if args.smoke:
        print("\n>>> Modo smoke: figuras NAO regeradas (usam os numeros R=100).")
        print(">>> Concluido. Saidas em results/*_SMOKE.txt")
    else:
        print("\n>>> Gerando figuras")
        import subprocess
        subprocess.run([sys.executable, str(HERE / "fig_generate.py")], check=False)
        print(">>> Concluido. Verifique results/ e figures/.")


if __name__ == "__main__":
    main()
