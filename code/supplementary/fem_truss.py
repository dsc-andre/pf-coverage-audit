"""
Caso de engenharia (Nivel 2): confiabilidade a fadiga de uma trelica plana.

FE real: trelica estaticamente INDETERMINADA (X-bracing). A distribuicao de
forcas depende das areas relativas dos membros -> o solve depende mesmo das
variaveis aleatorias (nao e statica trivial).

Variaveis aleatorias (5), amostradas em U-space (N(0,1)) e transformadas:
  u0 -> P       carga (lognormal)
  u1 -> A_chord area das cordas   (lognormal)
  u2 -> A_diag  area das diagonais(lognormal)
  u3 -> A_vert  area dos montantes(lognormal)
  u4 -> log_a   intercepto S-N    (normal)
Fadiga: Nf = 10^log_a / (Dsig[MPa])^m ;  estado-limite g = log10(Nf) - log10(Nreq).
g <= 0 => FALHA.
"""
import numpy as np

E_FIX = 210e9   # Pa (aco)
M_SN  = 3.0     # expoente S-N (fixo)
NREQ  = 5.0e5   # ciclos exigidos (ajustado -> Pf ~ 8e-3)

# --- geometria da trelica (cantilever, 2 vaos) ---
NODES = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.],[2.,0.],[2.,1.]])
MEMBERS = np.array([[0,2],[2,4],   # corda inferior  (grupo 0: chord)
                    [1,3],[3,5],   # corda superior  (grupo 0)
                    [2,3],[4,5],   # montantes       (grupo 2: vert)
                    [0,3],[2,5],   # diagonais       (grupo 1: diag)
                    [1,2],[3,4]])  # diagonais opostas (grupo 1) -> X-bracing
GROUP = np.array([0,0,0,0, 2,2, 1,1,1,1])   # 0=chord,1=diag,2=vert
FIXED = [0,1,2,3]                            # nos 0 e 1 engastados
LOAD_DOFS = [9, 11]                          # carga vertical nos nos 4 e 5

# pre-computa comprimentos e cossenos diretores
_VEC = NODES[MEMBERS[:,1]] - NODES[MEMBERS[:,0]]
_L   = np.hypot(_VEC[:,0], _VEC[:,1])
_COS = _VEC / _L[:,None]

def max_stress(P, areas3):
    """Maior |tensao| (Pa) entre os membros. areas3=[A_chord,A_diag,A_vert]."""
    A = areas3[GROUP]
    ndof = 2*len(NODES)
    K = np.zeros((ndof, ndof))
    for m,(i,j) in enumerate(MEMBERS):
        c,s = _COS[m]; kk = E_FIX*A[m]/_L[m]
        ke = kk*np.array([[c*c,c*s,-c*c,-c*s],
                          [c*s,s*s,-c*s,-s*s],
                          [-c*c,-c*s,c*c,c*s],
                          [-c*s,-s*s,c*s,s*s]])
        d=[2*i,2*i+1,2*j,2*j+1]
        K[np.ix_(d,d)] += ke
    F = np.zeros(ndof)
    for dd in LOAD_DOFS: F[dd] = -P
    free = [d for d in range(ndof) if d not in FIXED]
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free,free)], F[free])
    smax = 0.0
    for m,(i,j) in enumerate(MEMBERS):
        c,s = _COS[m]
        elong = (u[2*j]-u[2*i])*c + (u[2*j+1]-u[2*i+1])*s
        stress = E_FIX*elong/_L[m]            # = (E/L)*elong ; force/A = stress
        smax = max(smax, abs(stress))
    return smax

class TrussFatigue:
    name = "TrussFatigue(d=5)"
    d = 5
    # parametros das distribuicoes (lognormal: media, cov)
    P_m, P_cv   = 3.0e4, 0.15
    Ac_m, Ac_cv = 2.0e-3, 0.10
    Ad_m, Ad_cv = 1.2e-3, 0.10
    Av_m, Av_cv = 1.0e-3, 0.10
    loga_m, loga_s = 12.3, 0.20

    def __init__(self, rng):
        self.rng = rng
    def _ln(self, u, mean, cov):
        s = np.sqrt(np.log(1+cov**2)); mu = np.log(mean) - 0.5*s*s
        return np.exp(mu + s*u)
    def sample(self, n):
        return self.rng.standard_normal((n, self.d))
    def g(self, U):
        U = np.atleast_2d(U)
        P  = self._ln(U[:,0], self.P_m, self.P_cv)
        Ac = self._ln(U[:,1], self.Ac_m, self.Ac_cv)
        Ad = self._ln(U[:,2], self.Ad_m, self.Ad_cv)
        Av = self._ln(U[:,3], self.Av_m, self.Av_cv)
        loga = self.loga_m + self.loga_s*U[:,4]
        out = np.empty(len(U))
        for i in range(len(U)):
            smax = max_stress(P[i], np.array([Ac[i],Ad[i],Av[i]]))
            dsig = 2.0*smax/1e6                      # range em MPa
            Nf = 10**loga[i] / dsig**M_SN
            out[i] = np.log10(Nf) - np.log10(NREQ)
        return out
    def exact_pf(self):
        return None


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    prob = TrussFatigue(rng)
    # sanity 1: tensao deterministica no ponto medio (U=0)
    g0 = prob.g(np.zeros((1,5)))[0]
    print(f"g(U=0) = {g0:+.3f}  (g>0 => seguro no ponto medio)")
    s0 = max_stress(prob.P_m, np.array([prob.Ac_m, prob.Ad_m, prob.Av_m]))
    print(f"tensao max no ponto medio = {s0/1e6:.1f} MPa")
    # sanity 2: Pf por MCS
    U = prob.sample(200_000); g = prob.g(U)
    pf = np.mean(g <= 0)
    print(f"Pf (MCS 2e5) = {pf:.4e}")
    print(f"fracao com tensao > 0: ok={np.all(np.isfinite(g))}")
