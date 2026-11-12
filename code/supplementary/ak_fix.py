import numpy as np
import nivel1_benchmark as m

def ak_mcs_fixed(problem, pool, stats=None):
    """AK-MCS identico ao original, exceto: pontos ja avaliados sao excluidos da
    selecao (U=+inf). Sem isso, um ponto do pool pode ser re-selecionado, a matriz
    de covariancia fica singular e o ajuste do GP degenera."""
    Xd = problem.sample(m.N_INIT); yd = problem.g(Xd); n_calls = m.N_INIT
    used = np.zeros(len(pool), bool)
    while n_calls < m.N_BUDGET:
        gp = m.make_gp().fit(Xd, yd)
        mu, sd = gp.predict(pool, return_std=True); sd = np.maximum(sd, 1e-12)
        U = np.abs(mu)/sd; U[used] = np.inf
        i = int(np.argmin(U)); used[i] = True
        Xd = np.vstack([Xd, pool[i]]); yd = np.append(yd, problem.g(pool[i:i+1]))
        n_calls += 1
    gp = m.make_gp().fit(Xd, yd)
    mu, sd = gp.predict(pool, return_std=True); sd = np.maximum(sd, 1e-12)
    pf_hat = float(np.mean(mu <= 0))
    U = np.abs(mu)/sd
    K = min(m.AK_SIM_K, len(pool))
    idx = np.argsort(U)[:K]
    mu_s, cov_s = gp.predict(pool[idx], return_cov=True)
    jit = 1e-8
    for _ in range(6):
        try:
            L = np.linalg.cholesky(cov_s + jit*np.eye(K)); break
        except np.linalg.LinAlgError:
            jit *= 10
    else:
        L = np.diag(np.sqrt(np.maximum(np.diag(cov_s), 1e-12)))
    fixed_fail = int(np.sum(mu <= 0) - np.sum(mu[idx] <= 0))
    pf_real = np.empty(m.M_POST)
    for k in range(m.M_POST):
        g_real = mu_s + L @ m.RNG.standard_normal(K)
        pf_real[k] = (fixed_fail + int(np.sum(g_real <= 0))) / len(pool)
    lo, hi = np.percentile(pf_real, m.Q)
    if stats is not None:
        stats.append(dict(kernel=str(gp.kernel_), n_unc=int(np.sum(U<2))))
    return [("AK-MCS-fix", pf_hat, float(lo), float(hi))], n_calls

# --- variante de kernel para controle (ARD Matern 5/2) ---
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as Ck, Matern
def make_gp_ard(d):
    def f():
        return GaussianProcessRegressor(Ck(1.0,(1e-5,1e8))*Matern(np.ones(d),(1e-3,1e3),nu=2.5),
                                        normalize_y=True, n_restarts_optimizer=1, alpha=1e-10)
    return f
