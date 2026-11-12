"""CURVA ORCAMENTO x COBERTURA — versao FATIADA e RETOMAVEL.

A curva completa com a grade ampliada leva ~20 h. Este script a divide em
CELULAS independentes (um problema, um orcamento), que podem ser rodadas em
sessoes separadas, um pouco por dia. O progresso fica gravado, de modo que
nada se perde ao desligar a maquina.

CADA CELULA = (problema, orcamento). Sao 10 no total:
    problemas : truss (trelica FE), d10 (Linear d=10)
    orcamentos: 30, 50, 80, 120, 150

COMANDOS
    python run_curva_fatiada.py --status
        mostra o que ja foi feito e o que falta.

    python run_curva_fatiada.py --proximo
        roda a PROXIMA celula pendente (o mais simples: rode isso por dia).

    python run_curva_fatiada.py --problema truss --orcamento 80
        roda uma celula especifica.

    python run_curva_fatiada.py --tabela
        monta a tabela final com tudo que ja foi coletado.

    python run_curva_fatiada.py --smoke --problema truss --orcamento 30
        celula rapida de verificacao (R=5), gravada em arquivo separado.

ARQUIVOS
    ../results/curva_fatiada.json   progresso e metricas (nao apagar)
    ../results/curva_fatiada.txt    log legivel, em modo append

TEMPO POR CELULA (estimado a partir das medicoes anteriores, com a grade
ampliada e n_jobs=-1): cerca de 1 h para N=30, subindo para 3-4 h em N=150.
O script imprime o tempo real ao fim de cada celula, o que calibra as
seguintes.
"""
import argparse, json, pathlib, sys, time

import numpy as np
import nivel1_benchmark as m
import fem_truss as ft
import hiper

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)
JSON = RESULTS / "curva_fatiada.json"
LOG = RESULTS / "curva_fatiada.txt"

PROBLEMAS = ("truss", "d10")
ORCAMENTOS = (30, 50, 80, 120, 150)
R_PADRAO = 30            # suficiente para uma curva de tendencia


def carregar():
    if JSON.exists():
        return json.loads(JSON.read_text(encoding="utf-8"))
    return {}


def salvar(d):
    JSON.write_text(json.dumps(d, indent=2), encoding="utf-8")


def chave(prob, N):
    return f"{prob}_N{N}"


def pendentes(dados):
    return [(p, N) for N in ORCAMENTOS for p in PROBLEMAS
            if chave(p, N) not in dados]


def status(dados):
    print("=" * 74)
    print("  CURVA ORCAMENTO x COBERTURA — progresso")
    print("=" * 74)
    for N in ORCAMENTOS:
        linha = f"  N={N:<4}"
        for p in PROBLEMAS:
            ok = chave(p, N) in dados
            linha += f"  {p}: {'concluido' if ok else 'PENDENTE ':<10}"
        print(linha)
    falta = pendentes(dados)
    print("-" * 74)
    print(f"  {len(dados)} de {len(PROBLEMAS)*len(ORCAMENTOS)} celulas concluidas")
    if falta:
        p, N = falta[0]
        print(f"  proxima: problema={p}  orcamento={N}")
        print(f"  rode:    python run_curva_fatiada.py --proximo")
    else:
        print("  tudo concluido — rode --tabela para montar o resumo")
    print("=" * 74)


def montar_problema(nome):
    if nome == "truss":
        return ft.TrussFatigue(m.RNG)
    if nome == "d10":
        return m.Linear(beta=2.5, d=10)
    raise ValueError(nome)


def rodar_celula(prob, N, smoke=False):
    # configuracao identica a do estudo principal
    m.N_POOL, m.N_CAND, m.N_INIT = 40_000, 5_000, 12
    m.B_BOOT, m.B_AL, m.M_POST, m.AK_SIM_K = 200, 20, 200, 2000
    m.R_REPS = R_PADRAO
    m.METHODS = (m.ak_mcs, m.bootstrap_svr, m.conformal_svr)
    m.tune_svr = hiper.tune_svr           # grade ampliada
    m.N_BUDGET = N
    m.CONF_NCAL = max(20, int(round(0.4 * N)))

    if smoke:
        m.R_REPS, m.N_POOL, m.N_CAND = 5, 4000, 1000
        m.B_BOOT, m.B_AL, m.M_POST, m.AK_SIM_K = 40, 10, 60, 500
        print("*** MODO SMOKE — nao entra no progresso ***")

    print("=" * 74)
    print(f"  CELULA: problema={prob}  orcamento N_call={N}")
    print(f"  R_REPS={m.R_REPS}  conformal: {N - m.CONF_NCAL} treino "
          f"+ {m.CONF_NCAL} calibracao")
    print("=" * 74 + "\n")

    m.RNG = np.random.default_rng(2025)
    t0 = time.time()
    acc = m.run(montar_problema(prob))
    dt = (time.time() - t0) / 60
    print(f"\n>>> celula concluida em {dt:.1f} min")

    if smoke:
        return None

    resumo = {}
    for lab, d in acc.items():
        resumo[lab] = {
            "cobertura": 100 * d["cov"] / m.R_REPS,
            "cob_sup":   100 * d["up"] / m.R_REPS,
            "largura":   float(np.mean(d["w"])),
            "vies":      float(np.mean(d["bias"])),
            "int_score": float(np.mean(d["is"])),
        }
    return {"problema": prob, "orcamento": N, "R": m.R_REPS,
            "minutos": round(dt, 1), "metricas": resumo}


def escrever_log(cel):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n{'='*74}\nCELULA {cel['problema']} N={cel['orcamento']} "
                f"(R={cel['R']}, {cel['minutos']} min)\n")
        f.write(f"  {'metodo':<13}{'cobertura':>10}{'cob.sup':>9}"
                f"{'larg.media':>13}{'vies.medio':>13}{'int.score':>12}\n")
        for lab, v in cel["metricas"].items():
            f.write(f"  {lab:<13}{v['cobertura']:>9.0f}%{v['cob_sup']:>8.0f}%"
                    f"{v['largura']:>13.3e}{v['vies']:>+13.3e}"
                    f"{v['int_score']:>12.3e}\n")


def tabela(dados):
    metodos = ["AK-MCS", "BSVR-perc", "BSVR-BCa", "Conformal"]
    for prob in PROBLEMAS:
        print(f"\n### {prob} — cobertura (%) por orcamento")
        print("  " + "metodo".ljust(13) + "".join(f"{N:>8}" for N in ORCAMENTOS))
        for met in metodos:
            linha = "  " + met.ljust(13)
            for N in ORCAMENTOS:
                c = dados.get(chave(prob, N), {}).get("metricas", {}).get(met)
                linha += f"{c['cobertura']:>7.0f}%" if c else "       -"
            print(linha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--proximo", action="store_true")
    ap.add_argument("--tabela", action="store_true")
    ap.add_argument("--problema", choices=PROBLEMAS)
    ap.add_argument("--orcamento", type=int, choices=ORCAMENTOS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dados = carregar()

    if args.status:
        status(dados); return
    if args.tabela:
        tabela(dados); return

    if args.proximo:
        falta = pendentes(dados)
        if not falta:
            print("nada pendente — rode --tabela"); return
        prob, N = falta[0]
    elif args.problema and args.orcamento:
        prob, N = args.problema, args.orcamento
    else:
        status(dados)
        print("\nuse --proximo, ou --problema X --orcamento Y")
        return

    if not args.smoke and chave(prob, N) in dados:
        print(f"celula {chave(prob,N)} ja concluida; nada a fazer.")
        return

    cel = rodar_celula(prob, N, smoke=args.smoke)
    if cel is None:
        return

    dados[chave(prob, N)] = cel
    salvar(dados)
    escrever_log(cel)
    print(f">>> progresso salvo ({len(dados)}/"
          f"{len(PROBLEMAS)*len(ORCAMENTOS)} celulas)")
    falta = pendentes(dados)
    if falta:
        print(f">>> proxima: {falta[0][0]} N={falta[0][1]}  "
              f"(python run_curva_fatiada.py --proximo)")
    else:
        print(">>> todas as celulas concluidas — rode --tabela")


if __name__ == "__main__":
    main()
