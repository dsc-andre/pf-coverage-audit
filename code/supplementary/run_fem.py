"""Roda o benchmark de cobertura no caso de engenharia (trelica FE a fadiga).

Reproduz a Tabela 2 do artigo (TrussFatigue, d=5, R=100).
Tempo tipico: algumas horas em CPU de desktop.
"""
import numpy as np
import nivel1_benchmark as m
import fem_truss as ft

# ----------------------------------------------------------------------
# CONFIG DA RODADA FINAL DO ARTIGO — nao alterar para reproduzir a Tabela 2
# ----------------------------------------------------------------------
m.N_POOL   = 40_000
m.R_REPS   = 100        # rodada final do paper
m.B_BOOT   = 200
m.B_AL     = 20
m.M_POST   = 200
m.AK_SIM_K = 2000
m.N_CAND   = 5_000
m.N_INIT   = 12
m.N_BUDGET = 50
m.CONF_NCAL = 20
m.RNG = np.random.default_rng(2025)

truss = ft.TrussFatigue(m.RNG)

print("=" * 86)
print(f"  BENCHMARK FEM — {truss.name}")
print(f"  CONFIG DESTA RODADA:  R_REPS={m.R_REPS}  N_POOL={m.N_POOL}  "
      f"N_BUDGET={m.N_BUDGET}  B_BOOT={m.B_BOOT}")
if m.R_REPS < 100:
    print("  *** ATENCAO: R_REPS < 100 — NAO use esta rodada nas tabelas finais ***")
print("=" * 86 + "\n")

m.run(truss)
