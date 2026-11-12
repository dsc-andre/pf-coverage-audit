# pf-coverage-audit

Code and per-repetition results for the paper

> André Luiz Tod Emidio de Abreu. *Are surrogate-based intervals for the probability of failure trustworthy? A controlled coverage audit.* Submitted, 2026.

The paper measures how often the uncertainty intervals reported with a surrogate estimate of the probability of failure P_f actually contain the target, and by how much they miss when they do not. Four constructions are audited at a budget of fifty limit-state calls: the Kriging posterior interval of AK-MCS, the percentile and BCa bootstrap intervals of a support-vector surrogate, and a split-conformal bracket. Crude Monte Carlo with a Clopper–Pearson interval is the reference. The test problems are five analytical limit states and a finite-element truss fatigue problem.

## What is in this repository

```
code/main/            benchmark (v0.4) and the two runs behind the main tables
code/supplementary/   earlier benchmark version (v0.3) and the smaller, separate runs
results/main_run/     main run: every method, six problems, R = 100
results/control_C/    control on the upper bound of the SVR hyperparameter C
results/supplementary/  Kriging replication, kernel control, ablation on K
results/earlier_runs/   runs with the coarse and intermediate hyperparameter grids, budget curve
analysis/             scripts that turn the results into the tables and figures of the paper
```

Each line of a `*.jsonl` file is one repetition of one method (values rounded here):

```json
{"problem": "FourBranch(k=6.0)", "r": 7, "method": "BSVR-perc", "pf": 0.001425, "lo": 0.00125, "hi": 0.00183, "target": 0.00385}
```

`pf` is the point estimate, `[lo, hi]` the nominal 95% interval and `target` the failure fraction on the fixed Monte Carlo population (N = 40,000) that the surrogate is asked to reproduce.

## Reproducing the tables and figures from the stored results

This takes a few seconds and needs only `numpy`, `scipy` and `matplotlib`.

```bash
pip install -r requirements.txt
cd analysis
python aggregate.py   # results/ -> numbers.json (every number used in the paper)
python tables.py      # numbers.json -> tables.md (Tables 1 and 3 to 8)
python figures.py     # numbers.json -> figures/fig2_coverage, figures/fig3_budget
python tokens.py      # numbers.json -> tokens.json (numbers quoted in the running text)
```

`aggregate.py` recomputes every metric from the per-repetition files and checks the main run against the summary printed by the run itself (`results/main_run/v2.txt`). Figure 1 (the truss) is a schematic and is stored as an image.

## Re-running the experiments

The runs need `scikit-learn` in addition. Each script is run from its own folder.

| Script | What it produces | Cost |
|---|---|---|
| `code/main/run_v2.py` | main run, all methods on the same population (`--smoke` for a short test) | about 8.5 h on a laptop |
| `code/main/run_controle_C.py` | control on C, two paired arms, R = 50 | about 4 h |
| `code/supplementary/run_ak_fix.py` | Kriging-only runs, per problem and budget | depends on the problem and budget |

`run_v2.py` writes to a `results` folder next to its own folder; `run_controle_C.py` writes to `results_controle_C` beside the script; the supplementary scripts write to `out/` in the working directory. Seeds are fixed in the scripts (2025).

## Which file supports which result

| Item in the paper | Result files | Script |
|---|---|---|
| Tables 3, 4, 5 and 8; Figure 2 | `results/main_run/v2_reps.jsonl` | `code/main/run_v2.py` |
| Table 7 (control on C) | `results/control_C/` | `code/main/run_controle_C.py` |
| Table 6, coarse grid | `results/earlier_runs/coarse_grid/analytical_R100.txt`, `truss_R100.txt` | `code/supplementary/run_all.py` |
| Table 6, intermediate grid | `results/earlier_runs/controle_hiper.txt` | `code/supplementary/run_controle_hiper.py` |
| Conformal bracket under the coarse grid, replication | `results/earlier_runs/coarse_grid/controle_treino30.txt` | `code/supplementary/run_controle_b.py` |
| Kriging replication (Section 5.3) | `results/supplementary/ak_*_N50_s2025.jsonl` | `code/supplementary/run_ak_fix.py` |
| Kernel control on the truss | `results/supplementary/akard_*`, `akardrbf_*`, `akisomatern_*` | `run_ak_kernel.py`, `run_ak_kernel2.py` |
| Ablation on K | `results/supplementary/ablK_fix.jsonl` | `run_ablK_fix.py` |
| Figure 3 (budget curve) | `results/earlier_runs/curva_fatiada.json`, `results/supplementary/ak_{d10,truss}_N*.jsonl` | `run_curva_fatiada.py`, `run_ak_fix.py` |

`final_wide.txt`, `ablacao_K.txt`, `curva_fatiada.txt` and `svr2_*.jsonl` are earlier runs that the main run superseded. They are kept for the record and are still read by `aggregate.py`, but no number in the paper depends on them alone.

## Two versions of the benchmark

`code/supplementary/nivel1_benchmark.py` is version 0.3, with the coarse SVR grid and the strict tie treatment in BCa. Its AK-MCS could re-select a point already in the design, which made the Gaussian-process fit degenerate; `ak_fix.py` patches that, and every Kriging result in the paper comes from patched or later code. `results/supplementary/diag_bug_AK_trelica.log` shows the effect of the patch on the truss.

`code/main/nivel1_benchmark.py` is version 0.4: the patch is built in, the SVR grid is the extended one (`hiper.py`), BCa reports both tie treatments, the Monte Carlo reference is run on the same repetitions, and each repetition is logged.

## Reading the output

Code comments and the text summaries are in Portuguese. The column names in the `.txt` summaries are:

| Column | Meaning |
|---|---|
| `cob.` | coverage |
| `cob.sup` | upper coverage (upper bound at or above the target) |
| `sup.mat` | material upper coverage (upper bound at or above target − Monte Carlo standard error) |
| `falhas` / `mater.` | misses of the upper bound / material misses |
| `falta med.` / `falta max` | mean / maximum relative shortfall of the upper bound |
| `larg.` | interval width |
| `vies` | bias of the point estimate |
| `braco` | arm of the control (`base`: C ≤ 10⁵; `wide`: C ≤ 10⁷) |
| `trelica` | truss |

Method labels: `AK-MCS` (Kriging), `BSVR-perc` (percentile bootstrap), `BSVR-BCa` (BCa, mid-rank ties), `BCa-strict` (BCa, ties ignored), `Conformal` (split-conformal bracket), `MC-CP` (crude Monte Carlo with Clopper–Pearson).

## Licence

MIT, see `LICENSE`.
