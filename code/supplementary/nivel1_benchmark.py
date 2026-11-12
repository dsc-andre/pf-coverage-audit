"""
Benchmark de cobertura de Pf — nucleo do paper de diagnostico (v0.3).

Tese: os intervalos de confianca sobre a probabilidade de falha (Pf) de uso
corrente (Kriging-posteriori; bootstrap percentil; bootstrap BCa) SUB-COBREM
em estados-limite nao-lineares, embora fiquem proximos do nominal em casos
suaves. Auditamos isso de forma controlada (Pf verdadeira conhecida).

Upgrades vs v0.2:
  - AK-MCS com IC honesto: simulacao condicional CORRELACIONADA do Kriging
    na regiao da fronteira (substitui a aproximacao independente, que dava
    largura ~0). So os pontos incertos (menor U) sao simulados em conjunto.
  - Bateria de problemas: suave/nao-linear, baixa/alta dimensao, Pf comum/rara.
  - Pf verdadeira EXATA onde possivel (linear -> Phi; esfera -> qui-quadrado).

Autor: ABREU, A. L. E.
"""

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from scipy.stats import norm, chi2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
SEED      = 12345
N_POOL    = 40_000
R_REPS    = 100      # rodada final do paper (ja e o default p/ evitar retrabalho)
B_BOOT    = 200
B_AL      = 20
M_POST    = 200      # realizacoes condicionais do Kriging (IC do AK)
AK_SIM_K  = 2000     # nº de pontos de fronteira simulados em conjunto (AK)
N_CAND    = 5_000
N_INIT    = 12
N_BUDGET  = 50
CONF_NCAL = 20       # pontos de calibracao conformal (resto = treino)
NOMINAL   = 0.95
RNG       = np.random.default_rng(SEED)
Q         = [100*(1-NOMINAL)/2, 100*(1+NOMINAL)/2]


# ----------------------------------------------------------------------
# BATERIA DE PROBLEMAS.  Convencao: g(x) <= 0 => FALHA.
# ----------------------------------------------------------------------
class Linear:
    """g = beta - (sum x_i)/sqrt(d).  Suave.  Pf = Phi(-beta) EXATA."""
    def __init__(self, beta=2.5, d=2):
        self.beta, self.d, self.name = beta, d, f"Linear(b={beta},d={d})"
    def sample(self, n): return RNG.standard_normal((n, self.d))
    def g(self, X):      return self.beta - X.sum(1) / np.sqrt(self.d)
    def exact_pf(self):  return float(norm.cdf(-self.beta))

class Sphere:
    """g = r - ||x||, X~N(0,I_d).  Nao-linear (radial), suave.
       ||X||^2 ~ qui2_d -> Pf = P(qui2_d > r^2) EXATA."""
    def __init__(self, r=3.0, d=2):
        self.r, self.d, self.name = r, d, f"Sphere(r={r},d={d})"
    def sample(self, n): return RNG.standard_normal((n, self.d))
    def g(self, X):      return self.r - np.sqrt((X**2).sum(1))
    def exact_pf(self):  return float(chi2.sf(self.r**2, self.d))

class FourBranch:
    """Sistema em serie classico. Nao-linear, multi-regiao. Pf por MCS grande."""
    def __init__(self, k=6.0):
        self.k, self.d, self.name = k, 2, f"FourBranch(k={k})"
    def sample(self, n): return RNG.standard_normal((n, 2))
    def g(self, X):
        x1, x2 = X[:, 0], X[:, 1]
        b1 = 3 + 0.1*(x1-x2)**2 - (x1+x2)/np.sqrt(2)
        b2 = 3 + 0.1*(x1-x2)**2 + (x1+x2)/np.sqrt(2)
        b3 = (x1-x2) + self.k/np.sqrt(2)
        b4 = (x2-x1) + self.k/np.sqrt(2)
        return np.minimum.reduce([b1, b2, b3, b4])
    def exact_pf(self): return None


def build_pool(problem):
    pool = problem.sample(N_POOL)
    return pool, float(np.mean(problem.g(pool) <= 0))


# ----------------------------------------------------------------------
# AK-MCS  — IC via simulacao condicional correlacionada do Kriging
# ----------------------------------------------------------------------
def make_gp():
    return GaussianProcessRegressor(C(1.0,(1e-5,1e8))*RBF(1.0,(1e-3,1e3)),
                                    normalize_y=True, n_restarts_optimizer=1,
                                    alpha=1e-10)

def ak_mcs(problem, pool):
    Xd = problem.sample(N_INIT); yd = problem.g(Xd); n_calls = N_INIT
    while n_calls < N_BUDGET:
        gp = make_gp().fit(Xd, yd)
        mu, sd = gp.predict(pool, return_std=True); sd = np.maximum(sd, 1e-12)
        i = int(np.argmin(np.abs(mu)/sd))
        Xd = np.vstack([Xd, pool[i]]); yd = np.append(yd, problem.g(pool[i:i+1]))
        n_calls += 1
    gp = make_gp().fit(Xd, yd)
    mu, sd = gp.predict(pool, return_std=True); sd = np.maximum(sd, 1e-12)
    pf_hat = float(np.mean(mu <= 0))

    # subconjunto incerto -> simulacao condicional CORRELACIONADA
    U = np.abs(mu)/sd
    K = min(AK_SIM_K, len(pool))
    idx = np.argsort(U)[:K]
    mu_s, cov_s = gp.predict(pool[idx], return_cov=True)
    jit = 1e-8
    for _ in range(6):
        try:
            L = np.linalg.cholesky(cov_s + jit*np.eye(K)); break
        except np.linalg.LinAlgError:
            jit *= 10
    else:
        L = np.diag(np.sqrt(np.maximum(np.diag(cov_s), 1e-12)))  # fallback
    fixed_fail = int(np.sum(mu <= 0) - np.sum(mu[idx] <= 0))
    pf_real = np.empty(M_POST)
    for m in range(M_POST):
        g_real = mu_s + L @ RNG.standard_normal(K)
        pf_real[m] = (fixed_fail + int(np.sum(g_real <= 0))) / len(pool)
    lo, hi = np.percentile(pf_real, Q)
    return [("AK-MCS", pf_hat, float(lo), float(hi))], n_calls


# ----------------------------------------------------------------------
# Bootstrap-SVR (aprendizado ativo por spread; IC percentil + BCa)
# ----------------------------------------------------------------------
def tune_svr(Xs, y):
    grid = {"C":[1,10,100], "gamma":["scale",0.1,1.0], "epsilon":[0.01,0.1]}
    return GridSearchCV(SVR(kernel="rbf"), grid, cv=3,
                        scoring="neg_mean_squared_error").fit(Xs, y).best_params_

def _pf(Xtr, ytr, params, pool_s):
    return float(np.mean(SVR(kernel="rbf", **params).fit(Xtr, ytr)
                         .predict(pool_s) <= 0))

def bca(boot, theta, jack, alpha):
    B = len(boot)
    z0 = norm.ppf(np.clip(np.mean(boot < theta), 1/(B+1), 1-1/(B+1)))
    diff = np.mean(jack) - jack; s2 = np.sum(diff**2)
    a = np.sum(diff**3)/(6*s2**1.5) if s2 > 0 else 0.0
    adj = lambda z: norm.cdf(z0 + (z0+z)/(1 - a*(z0+z)))
    a1, a2 = np.clip([adj(norm.ppf(alpha/2)), adj(norm.ppf(1-alpha/2))], 1e-4, 1-1e-4)
    return float(np.percentile(boot, 100*a1)), float(np.percentile(boot, 100*a2))

def bootstrap_svr(problem, pool):
    Xd = problem.sample(N_INIT); yd = problem.g(Xd); n_calls = N_INIT
    cand = pool[RNG.choice(len(pool), N_CAND, replace=False)]
    while n_calls < N_BUDGET:
        sc = StandardScaler().fit(Xd); Xs = sc.transform(Xd)
        params = tune_svr(Xs, yd); cs = sc.transform(cand)
        P = np.empty((B_AL, len(cand)))
        for b in range(B_AL):
            ix = RNG.integers(0, len(yd), len(yd))
            P[b] = SVR(kernel="rbf", **params).fit(Xs[ix], yd[ix]).predict(cs)
        i = int(np.argmin(np.abs(P.mean(0))/np.maximum(P.std(0),1e-12)))
        Xd = np.vstack([Xd, cand[i]]); yd = np.append(yd, problem.g(cand[i:i+1]))
        n_calls += 1
    sc = StandardScaler().fit(Xd); Xs = sc.transform(Xd)
    params = tune_svr(Xs, yd); pool_s = sc.transform(pool)
    pf_orig = _pf(Xs, yd, params, pool_s)
    n = len(yd); boot = np.empty(B_BOOT)
    for b in range(B_BOOT):
        ix = RNG.integers(0, n, n); boot[b] = _pf(Xs[ix], yd[ix], params, pool_s)
    jack = np.empty(n)
    for i in range(n):
        msk = np.ones(n, bool); msk[i] = False
        s2 = StandardScaler().fit(Xd[msk])
        jack[i] = _pf(s2.transform(Xd[msk]), yd[msk], params, s2.transform(pool))
    lo_p, hi_p = np.percentile(boot, Q)
    lo_b, hi_b = bca(boot, pf_orig, jack, 1-NOMINAL)
    return [("BSVR-perc", pf_orig, float(lo_p), float(hi_p)),
            ("BSVR-BCa",  pf_orig, lo_b, hi_b)], n_calls


# ----------------------------------------------------------------------
# Conformal-SVR (o conserto) — split conformal sobre g, propagado p/ Pf.
# Orcamento: N_BUDGET - CONF_NCAL em treino (aprendizado ativo) +
# CONF_NCAL pontos i.i.d. de calibracao (trocaveis com o pool).
# IC sobre Pf: bracket [certamente-falha, pode-falhar].
# Se o surrogate erra muito, os residuos crescem, q cresce e a faixa
# alarga sozinha (o "nao sei" honesto). Robusto a vies por construcao.
# ----------------------------------------------------------------------
def conformal_svr(problem, pool):
    n_train = N_BUDGET - CONF_NCAL
    Xd = problem.sample(N_INIT); yd = problem.g(Xd); n_calls = N_INIT
    cand = pool[RNG.choice(len(pool), N_CAND, replace=False)]
    while n_calls < n_train:                       # aprendizado ativo p/ treino
        sc = StandardScaler().fit(Xd); Xs = sc.transform(Xd)
        params = tune_svr(Xs, yd); cs = sc.transform(cand)
        P = np.empty((B_AL, len(cand)))
        for b in range(B_AL):
            ix = RNG.integers(0, len(yd), len(yd))
            P[b] = SVR(kernel="rbf", **params).fit(Xs[ix], yd[ix]).predict(cs)
        i = int(np.argmin(np.abs(P.mean(0))/np.maximum(P.std(0),1e-12)))
        Xd = np.vstack([Xd, cand[i]]); yd = np.append(yd, problem.g(cand[i:i+1]))
        n_calls += 1
    sc = StandardScaler().fit(Xd); Xs = sc.transform(Xd)
    params = tune_svr(Xs, yd)
    svr = SVR(kernel="rbf", **params).fit(Xs, yd)
    # calibracao: pontos frescos i.i.d. da distribuicao de entrada
    Xc = problem.sample(CONF_NCAL); gc = problem.g(Xc)
    res = np.abs(gc - svr.predict(sc.transform(Xc)))
    k = int(np.ceil((CONF_NCAL + 1) * NOMINAL))    # quantil conformal
    q = np.inf if k > CONF_NCAL else float(np.sort(res)[k-1])
    g_pool = svr.predict(sc.transform(pool))
    pf_pt = float(np.mean(g_pool <= 0))
    pf_lo = float(np.mean(g_pool + q <= 0))        # certamente falha
    pf_hi = float(np.mean(g_pool - q <= 0))        # pode falhar
    return [("Conformal", pf_pt, pf_lo, pf_hi)], N_BUDGET


# ----------------------------------------------------------------------
# [ABLACAO — NAO faz parte dos resultados principais do paper]
# Conformal-CV com escores normalizados. Testado e DESCARTADO como metodo:
# a normalizacao pelo spread dos fold-modelos reintroduz a dependencia do
# CONSENSO do ensemble — quando os modelos erram juntos (alta dim), a banda
# colapsa e a cobertura cai (ver Discussao). Mantido aqui so para reproduzir
# esse resultado negativo. Para rodar a ablacao:
#   m.METHODS = (conformal_svr, conformal_cv_svr); m.run(problema)
# ----------------------------------------------------------------------
CONF_K = 5        # folds sobre o conjunto de calibracao
S_FLOOR = 1e-6    # piso do normalizador

def conformal_cv_svr(problem, pool):
    n_train = N_BUDGET - CONF_NCAL
    Xd = problem.sample(N_INIT); yd = problem.g(Xd); n_calls = N_INIT
    cand = pool[RNG.choice(len(pool), N_CAND, replace=False)]
    while n_calls < n_train:
        sc = StandardScaler().fit(Xd); Xs = sc.transform(Xd)
        params = tune_svr(Xs, yd); cs = sc.transform(cand)
        P = np.empty((B_AL, len(cand)))
        for b in range(B_AL):
            ix = RNG.integers(0, len(yd), len(yd))
            P[b] = SVR(kernel="rbf", **params).fit(Xs[ix], yd[ix]).predict(cs)
        i = int(np.argmin(np.abs(P.mean(0))/np.maximum(P.std(0),1e-12)))
        Xd = np.vstack([Xd, cand[i]]); yd = np.append(yd, problem.g(cand[i:i+1]))
        n_calls += 1
    Xc = problem.sample(CONF_NCAL); gc = problem.g(Xc)

    folds = np.array_split(RNG.permutation(CONF_NCAL), CONF_K)
    Gpool = np.empty((CONF_K, len(pool)))   # predicoes dos fold-modelos
    Gcal  = np.empty((CONF_K, CONF_NCAL))
    for k, fold in enumerate(folds):
        msk = np.ones(CONF_NCAL, bool); msk[fold] = False
        Xtr = np.vstack([Xd, Xc[msk]]); ytr = np.concatenate([yd, gc[msk]])
        sck = StandardScaler().fit(Xtr); Xtrs = sck.transform(Xtr)
        pk = tune_svr(Xtrs, ytr)
        mk = SVR(kernel="rbf", **pk).fit(Xtrs, ytr)
        Gpool[k] = mk.predict(sck.transform(pool))
        Gcal[k]  = mk.predict(sck.transform(Xc))
    s_raw_cal  = Gcal.std(0)
    s_raw_pool = Gpool.std(0)
    # residuo out-of-fold bruto de cada ponto de calibracao
    r_oof = np.empty(CONF_NCAL)
    for k, fold in enumerate(folds):
        r_oof[fold] = np.abs(gc[fold] - Gcal[k][fold])
    # piso aditivo: mediana dos residuos oof -> impede o colapso da banda
    # quando os fold-modelos erram juntos (consenso != acerto)
    floor = max(float(np.median(r_oof)), S_FLOOR)
    s_cal  = s_raw_cal  + floor
    s_pool = s_raw_pool + floor
    z = r_oof / s_cal
    kq = int(np.ceil((CONF_NCAL + 1) * NOMINAL))
    qz = np.inf if kq > CONF_NCAL else float(np.sort(z)[kq-1])
    g_hat = Gpool.mean(0)
    band  = qz * s_pool
    pf_pt = float(np.mean(g_hat <= 0))
    pf_lo = float(np.mean(g_hat + band <= 0))
    pf_hi = float(np.mean(g_hat - band <= 0))
    return [("Conf-CVnorm", pf_pt, pf_lo, pf_hi)], N_BUDGET


# ----------------------------------------------------------------------
# LACO DE COBERTURA
# ----------------------------------------------------------------------
# metodos dos resultados principais (CVnorm fica de fora — ver ablacao acima)
METHODS = (ak_mcs, bootstrap_svr, conformal_svr)

def run(problem):
    pool, pf_pool = build_pool(problem)
    exact = problem.exact_pf(); target = pf_pool
    alpha = 1 - NOMINAL
    acc = {}
    print(f"  rodando {problem.name} ...", flush=True)
    for r in range(R_REPS):
        for fn in METHODS:
            entries, calls = fn(problem, pool)
            for lab, pf, lo, hi in entries:
                d = acc.setdefault(lab, {"cov":0,"up":0,"w":[],"bias":[],"is":[]})
                d["cov"] += int(lo <= target <= hi)
                d["up"]  += int(hi >= target)          # limite superior cobre?
                wid = hi - lo
                isc = wid + (2/alpha)*max(lo-target,0) + (2/alpha)*max(target-hi,0)
                d["w"].append(wid); d["bias"].append(pf-target); d["is"].append(isc)
        print(f"\r    repeticao {r+1}/{R_REPS}", end="", flush=True)
    print()
    print(f"{'-'*86}\nPROBLEMA: {problem.name}")
    print(f"  Pf(pool N={N_POOL}) = {pf_pool:.4e}"
          + (f" | exata = {exact:.4e}" if exact is not None else ""))
    print(f"  {'metodo':<13}{'cobertura':>10}{'cob.sup':>9}{'larg.media':>13}"
          f"{'vies.medio':>13}{'int.score':>12}")
    for lab, d in acc.items():
        print(f"  {lab:<13}{d['cov']/R_REPS:>10.0%}{d['up']/R_REPS:>9.0%}"
              f"{np.mean(d['w']):>13.3e}{np.mean(d['bias']):>+13.3e}"
              f"{np.mean(d['is']):>12.3e}")
    return acc


def main():
    print("="*86)
    print(f"  CONFIG DESTA RODADA:  R_REPS={R_REPS}  N_POOL={N_POOL}  "
          f"N_BUDGET={N_BUDGET}  B_BOOT={B_BOOT}")
    if R_REPS < 100:
        print("  *** ATENCAO: R_REPS < 100 — NAO use esta rodada nas tabelas finais ***")
    print("="*86 + "\n")
    bateria = [
        Linear(beta=2.33, d=2),    # suave, Pf ~ 1e-2
        Linear(beta=3.09, d=2),    # suave, Pf ~ 1e-3 (varredura: mais raro)
        Linear(beta=2.5,  d=10),   # suave, alta dimensao
        Sphere(r=3.0,     d=2),    # nao-linear (radial), suave
        FourBranch(k=6.0),         # nao-linear, multi-regiao (o caso dificil)
    ]
    for p in bateria:
        run(p)
    print(f"\n{'='*86}\nLeitura: cobertura ~91-99% = calibrado (R=100); int.score "
          "menor = melhor\n(largura + multa por nao cobrir); cob.sup = limite "
          "superior cobre Pf (seguranca).")


if __name__ == "__main__":
    main()
