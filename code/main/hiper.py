"""Ajuste de hiperparametros do SVR — grade ampliada com diagnostico de borda.

MOTIVACAO
O protocolo original buscava C em {1,10,100} e gamma em {"scale",0.1,1.0}. O
controle de hiperparametros mostrou que o otimo cai na BORDA SUPERIOR dessa
grade, de modo que o SVR ficava subajustado e o vies da estimativa de Pf era
uma a duas ordens de grandeza maior do que precisava ser (trelica: -4,4e-3 com
a grade estreita contra -1,8e-4 com a ampliada).

JUSTIFICATIVA DE PRINCIPIO (nao e "ampliar ate funcionar")
A funcao de estado-limite e DETERMINISTICA: nao ha ruido de observacao. Em
experimentos computacionais sem ruido, o ajuste correto e de regularizacao
fraca — C grande e epsilon pequeno —, aproximando-se da interpolacao. E a mesma
razao pela qual o Kriging e usado aqui como INTERPOLADOR e nao como suavizador.
A grade abaixo cobre esse regime em escala logaritmica.

DIAGNOSTICO DE BORDA
Toda vez que o otimo selecionado cai na extremidade da grade, o fato e
registrado; `border_report()` informa a fracao de ocorrencias.

ATENCAO NA LEITURA: borda superior de C baixa (~2%) indica teto folgado. Ja o
otimo de gamma e epsilon cai FREQUENTEMENTE no minimo da grade, e isso NAO
significa truncamento: verificamos que abaixo de gamma=1e-5 o modelo DEGENERA —
com gamma=1e-7 o erro quadratico de treino sobe de 2e-4 para 6e-2 e a estimativa
de Pf colapsa para zero. Isto e, 1e-5 nao e um limite arbitrario que estamos
evitando ultrapassar: e o fim do regime util, e uma grade mais ampla apenas
ofereceria modelos piores, que a validacao cruzada nao escolheria. O que a
selecao indica e que, com poucos pontos, o ajuste otimo e o de kernel quase
plano — proximo de um SVR linear, que nestes problemas reproduz bem a Pf.

USO
    import hiper, nivel1_benchmark as m
    m.tune_svr = hiper.tune_svr          # troca o ajustador
    ...                                   # roda o benchmark normalmente
    print(hiper.border_report())
"""
from collections import Counter

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# grade log-uniforme ampla; regularizacao fraca incluida (problema sem ruido)
GRID = {
    "C":       [1e0, 1e1, 1e2, 1e3, 1e4, 1e5],
    "gamma":   ["scale", 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
    "epsilon": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
}

_EDGES = {
    "C":       {min(GRID["C"]), max(GRID["C"])},
    "gamma":   {1e-5, 1e0},            # "scale" nao e extremidade ordenada
    "epsilon": {min(GRID["epsilon"]), max(GRID["epsilon"])},
}

_counts = Counter()


def tune_svr(Xs, y):
    """Igual a versao original, porem com grade ampliada e registro de borda."""
    best = GridSearchCV(SVR(kernel="rbf"), GRID, cv=3,
                        scoring="neg_mean_squared_error",
                        n_jobs=-1).fit(Xs, y).best_params_
    _counts["total"] += 1
    for k, edges in _EDGES.items():
        if best[k] in edges:
            _counts[k] += 1
            lado = "max" if best[k] == max(edges) else "min"
            _counts[f"{k}_{lado}"] += 1
    # v0.4: C no teto JUNTO com gamma no piso e o regime quase-linear (crista
    # C*gamma ~ constante), nao truncamento: registra-se a parte.
    if best["C"] == max(GRID["C"]) and best["gamma"] == 1e-5:
        _counts["crista"] += 1
    return best


def reset_counts():
    _counts.clear()


def border_report():
    """Resumo textual da frequencia com que o otimo caiu na borda da grade."""
    n = _counts.get("total", 0)
    if not n:
        return "diagnostico de borda: nenhuma chamada registrada."
    linhas = [f"diagnostico de borda ({n} ajustes):"]
    for k in ("C", "gamma", "epsilon"):
        c = _counts.get(k, 0)
        lo, hi = _counts.get(f"{k}_min", 0), _counts.get(f"{k}_max", 0)
        linhas.append(f"  {k:<8} otimo na borda em {c:5d} ({100*c/n:5.1f}%)"
                      f"  [min: {lo}, max: {hi}]")
    cr = _counts.get("crista", 0)
    linhas.append(f"  C no teto E gamma no piso simultaneamente: {cr} ({100*cr/n:.1f}%)"
                  " -> regime quase-linear, nao truncamento")
    linhas.append("  (fracoes altas indicam grade ainda truncada)")
    return "\n".join(linhas)
