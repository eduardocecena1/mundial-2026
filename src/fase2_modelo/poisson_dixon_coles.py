"""
poisson_dixon_coles.py — FASE 2: motor principal de predicción de goles.

Implementa el modelo de Dixon-Coles (1997), una extensión del modelo de Poisson
bivariado pensada justamente para fútbol, con una capa JERÁRQUICA por liga que
es lo que lo hace utilizable en fútbol de clubes:

  goles_local ~ Poisson(lambda),  log(lambda) = c + ventaja[liga_partido]*(no_neutral)
                                                  + ataque[local] - defensa[visit]
  goles_visit ~ Poisson(mu),      log(mu)     = c + ataque[visit] - defensa[local]

  ataque[i]  = A[liga(i)] + a[i]        <- efecto de liga + desviación del equipo
  defensa[i] = D[liga(i)] + d[i]

Por qué la capa de liga. Un Dixon-Coles plano estima una fuerza por equipo y la
encoge hacia la media GLOBAL. Con 15 ligas de nivel muy distinto conectadas solo
por las copas europeas, eso sobrevalora a los equipos de ligas débiles (que ganan
casi todos sus partidos domésticos) y castiga a los de ligas fuertes. Con la capa
jerárquica, cada equipo se encoge hacia la media de SU liga, y la diferencia de
nivel entre ligas la estiman los partidos UEFA, que son los únicos que las
comparan directamente.

Mejoras sobre el Poisson simple, todas orientadas a PRECISIÓN:

  1. Corrección de Dixon-Coles (parámetro rho) para los marcadores bajos
     (0-0, 1-0, 0-1, 1-1), que el Poisson puro estima mal.
  2. Verosimilitud PONDERADA:
       - decaimiento temporal: los partidos recientes pesan más (vida media en config).
       - peso por competición: una eliminatoria de Champions informa más que una
         previa de la Conference.
  3. Ventaja de local POR LIGA: varía mucho entre ligas (y cayó tras 2020). Además
     solo se aplica si no es campo neutral.
  4. Regularización L2 (ridge) en dos niveles: fuerte sobre la desviación del
     equipo respecto a su liga, suave sobre el efecto de liga (que tiene datos de
     sobra). Implementa el requisito de "no inventar números".

Se entrena por máxima verosimilitud con gradiente ANALÍTICO (L-BFGS-B), lo que
permite ajustar cientos de equipos y decenas de miles de partidos en segundos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

# Clave de liga que se usa cuando no se conoce la de un equipo o competición.
LIGA_POR_DEFECTO = "otras"


# --- Pesos (decaimiento temporal + competición) ------------------------------

def peso_torneo(competicion: str, pesos: dict) -> float:
    """Peso de importancia de una competición (una previa informa menos que la UCL).

    `pesos` es el bloque `datos.competiciones` de config.yaml, donde cada entrada
    es {peso: .., liga: ..}. Se acepta también un valor suelto por comodidad."""
    meta = pesos.get(competicion)
    if isinstance(meta, dict):
        return float(meta.get("peso", 0.70))
    if meta is not None:
        return float(meta)
    return 0.70


def peso_temporal(edad_anios: np.ndarray, vida_media_anios: float) -> np.ndarray:
    """Decaimiento exponencial: w = 0.5 ** (edad / vida_media)."""
    xi = math.log(2.0) / max(vida_media_anios, 1e-6)
    return np.exp(-xi * np.clip(edad_anios, 0.0, None))


# --- El modelo --------------------------------------------------------------

@dataclass
class DixonColes:
    max_goles: int = 8
    reg: float = 1.0                  # ridge sobre la desviación del equipo
    reg_liga: float = 0.05            # ridge (suave) sobre el efecto de liga
    # rellenados tras fit():
    equipos: list = field(default_factory=list)
    idx: dict = field(default_factory=dict)
    ligas: list = field(default_factory=list)
    idx_liga: dict = field(default_factory=dict)
    liga_equipo: dict = field(default_factory=dict)   # nombre equipo -> clave de liga
    ataque: np.ndarray = None         # A[liga] + a[equipo] (fuerza total)
    defensa: np.ndarray = None
    a_equipo: np.ndarray = None       # solo la desviación respecto a su liga
    d_equipo: np.ndarray = None
    A_liga: np.ndarray = None         # efecto de liga en ataque
    D_liga: np.ndarray = None
    G_liga: np.ndarray = None         # ventaja de local por liga
    intercepto: float = 0.0
    rho: float = 0.0
    liga_defecto: str = LIGA_POR_DEFECTO
    _ajustado: bool = False

    # ---- Entrenamiento ----

    def fit(self, local, visit, gl, gv, neutral, peso,
            liga_equipo: dict | None = None, liga_partido=None):
        """Ajusta los parámetros por máxima verosimilitud ponderada.

        Args (arrays de igual longitud, una entrada por partido jugado):
          local, visit  : nombres de equipo (local y visitante)
          gl, gv        : goles de local y visitante (int)
          neutral       : 1 si campo neutral, 0 si hay localía real
          peso          : peso del partido (decaimiento temporal * peso competición)
          liga_equipo   : dict nombre de equipo -> clave de liga. Si es None, todos
                          los equipos caen en una sola liga y el modelo se reduce
                          al Dixon-Coles plano de siempre (modo Mundial).
          liga_partido  : clave de liga de cada partido, para la ventaja de local.
                          Si es None, se usa la liga del equipo local.
        """
        local = np.asarray(local)
        visit = np.asarray(visit)
        gl = np.asarray(gl, dtype=float)
        gv = np.asarray(gv, dtype=float)
        neutral = np.asarray(neutral, dtype=float)
        w = np.asarray(peso, dtype=float)

        # Índice de equipos
        self.equipos = sorted(set(local) | set(visit))
        self.idx = {e: i for i, e in enumerate(self.equipos)}
        n = len(self.equipos)
        ih = np.array([self.idx[e] for e in local])
        ia = np.array([self.idx[e] for e in visit])

        # Índice de ligas: la de cada equipo y la de cada partido
        liga_equipo = liga_equipo or {}
        self.liga_equipo = {e: liga_equipo.get(e, LIGA_POR_DEFECTO) for e in self.equipos}
        if liga_partido is None:
            liga_partido = [self.liga_equipo[e] for e in local]
        liga_partido = np.asarray(liga_partido)

        self.ligas = sorted(set(self.liga_equipo.values()) | set(liga_partido.tolist()))
        self.idx_liga = {l: i for i, l in enumerate(self.ligas)}
        L = len(self.ligas)
        le = np.array([self.idx_liga[self.liga_equipo[e]] for e in self.equipos])  # liga de cada equipo
        lp = np.array([self.idx_liga[l] for l in liga_partido])                    # liga de cada partido

        # Máscaras de los marcadores bajos para la corrección de Dixon-Coles
        m00 = (gl == 0) & (gv == 0)
        m01 = (gl == 0) & (gv == 1)
        m10 = (gl == 1) & (gv == 0)
        m11 = (gl == 1) & (gv == 1)
        no_neutral = 1.0 - neutral

        # Vector de parámetros: [a(n), d(n), A(L), D(L), G(L), intercepto, rho]
        o_a, o_d, o_A, o_D, o_G = 0, n, 2 * n, 2 * n + L, 2 * n + 2 * L
        o_c, o_rho = 2 * n + 3 * L, 2 * n + 3 * L + 1
        n_par = o_rho + 1

        def desempaqueta(p):
            return (p[o_a:o_a + n], p[o_d:o_d + n],
                    p[o_A:o_A + L], p[o_D:o_D + L], p[o_G:o_G + L],
                    p[o_c], p[o_rho])

        def nll_y_grad(p):
            a, d, A, D, G, c, rho = desempaqueta(p)
            atk = A[le] + a          # fuerza total por equipo
            dfn = D[le] + d

            eta1 = c + G[lp] * no_neutral + atk[ih] - dfn[ia]
            eta2 = c + atk[ia] - dfn[ih]
            lam = np.exp(np.clip(eta1, -10, 10))
            mu = np.exp(np.clip(eta2, -10, 10))

            # Log-verosimilitud Poisson (sin el término factorial, constante)
            ll = gl * eta1 - lam + gv * eta2 - mu

            # Corrección Dixon-Coles tau sobre los 4 marcadores bajos
            tau = np.ones_like(lam)
            tau = np.where(m00, 1.0 - lam * mu * rho, tau)
            tau = np.where(m01, 1.0 + lam * rho, tau)
            tau = np.where(m10, 1.0 + mu * rho, tau)
            tau = np.where(m11, 1.0 - rho, tau)
            tau = np.clip(tau, 1e-10, None)
            ll = ll + np.log(tau)

            nll = (-np.sum(w * ll)
                   + self.reg * (np.sum(a ** 2) + np.sum(d ** 2))
                   + self.reg_liga * (np.sum(A ** 2) + np.sum(D ** 2)))

            # --- Gradiente analítico ---
            # d ll / d eta1 (Poisson) = gl - lam ; idem eta2
            # d log(tau)/d eta1 = (1/tau) * d tau/d lam * lam
            dtau_dlam = np.zeros_like(lam)
            dtau_dmu = np.zeros_like(lam)
            dtau_drho = np.zeros_like(lam)
            dtau_dlam = np.where(m00, -mu * rho, dtau_dlam)
            dtau_dmu = np.where(m00, -lam * rho, dtau_dmu)
            dtau_drho = np.where(m00, -lam * mu, dtau_drho)
            dtau_dlam = np.where(m01, rho, dtau_dlam)
            dtau_drho = np.where(m01, lam, dtau_drho)
            dtau_dmu = np.where(m10, rho, dtau_dmu)
            dtau_drho = np.where(m10, mu, dtau_drho)
            dtau_drho = np.where(m11, -1.0, dtau_drho)

            g_eta1 = w * (gl - lam + (dtau_dlam / tau) * lam)
            g_eta2 = w * (gv - mu + (dtau_dmu / tau) * mu)
            g_rho = np.sum(w * (dtau_drho / tau))

            # Estos son gradientes de la LOG-verosimilitud; la NLL invierte el signo.
            grad = np.zeros(n_par)

            # Desviaciones por equipo (acumulación por equipo)
            g_atk = np.zeros(n)
            g_dfn = np.zeros(n)
            np.add.at(g_atk, ih, -g_eta1)                # d eta1/d atk[local]  = +1
            np.add.at(g_atk, ia, -g_eta2)                # d eta2/d atk[visit]  = +1
            np.add.at(g_dfn, ia, +g_eta1)                # d eta1/d defensa[visit] = -1
            np.add.at(g_dfn, ih, +g_eta2)                # d eta2/d defensa[local] = -1
            grad[o_a:o_a + n] = g_atk
            grad[o_d:o_d + n] = g_dfn

            # Efectos de liga: la fuerza de un equipo entra como A[liga]+a, así que
            # el gradiente de A[l] es la suma de los de sus equipos.
            grad[o_A:o_A + L] = np.bincount(le, weights=g_atk, minlength=L)
            grad[o_D:o_D + L] = np.bincount(le, weights=g_dfn, minlength=L)

            # Ventaja de local, por liga del partido
            grad[o_G:o_G + L] = np.bincount(lp, weights=-g_eta1 * no_neutral, minlength=L)

            grad[o_c] = -np.sum(g_eta1 + g_eta2)         # intercepto
            grad[o_rho] = -g_rho                          # rho

            # Término de regularización
            grad[o_a:o_a + n] += 2 * self.reg * a
            grad[o_d:o_d + n] += 2 * self.reg * d
            grad[o_A:o_A + L] += 2 * self.reg_liga * A
            grad[o_D:o_D + L] += 2 * self.reg_liga * D
            return nll, grad

        # Inicialización y límites
        p0 = np.zeros(n_par)
        p0[o_c] = math.log(max(np.average(gl + gv, weights=w) / 2.0, 0.1))  # intercepto
        p0[o_G:o_G + L] = 0.25                                              # ventaja local
        p0[o_rho] = -0.05
        limites = ([(-3, 3)] * (2 * n)                 # a, d
                   + [(-1.5, 1.5)] * (2 * L)           # A, D
                   + [(-0.5, 1.0)] * L                 # G
                   + [(-1, 2), (-0.2, 0.2)])           # intercepto, rho

        res = minimize(nll_y_grad, p0, jac=True, method="L-BFGS-B",
                       bounds=limites, options={"maxiter": 2000, "ftol": 1e-9})

        a, d, A, D, G, c, rho = desempaqueta(res.x)
        self.a_equipo, self.d_equipo = a, d
        self.A_liga, self.D_liga, self.G_liga = A, D, G
        # Fuerza total por equipo: lo que consumen los mercados y el ranking.
        self.ataque = A[le] + a
        self.defensa = D[le] + d
        self.intercepto, self.rho = c, rho
        self._ajustado = True
        return self

    # ---- Acceso a parámetros ----

    @property
    def ventaja_local(self) -> float:
        """Ventaja de local media (para informes). La real depende de la liga."""
        if self.G_liga is None:
            return 0.0
        return float(np.mean(self.G_liga))

    def ventaja_de_liga(self, liga: str | None) -> float:
        """Ventaja de local de una liga concreta, con caída elegante a la media."""
        if self.G_liga is None:
            return 0.0
        i = self.idx_liga.get(liga if liga else self.liga_defecto)
        return float(self.G_liga[i]) if i is not None else float(np.mean(self.G_liga))

    def fuerza_ligas(self) -> dict[str, dict]:
        """Efectos estimados por liga: ataque, defensa y ventaja de local."""
        if not self._ajustado:
            return {}
        return {l: {"ataque": float(self.A_liga[i]),
                    "defensa": float(self.D_liga[i]),
                    "ventaja_local": float(self.G_liga[i])}
                for l, i in self.idx_liga.items()}

    # ---- Predicción ----

    def _lambdas(self, local: str, visit: str, neutral: int, liga: str | None = None):
        """Goles esperados (lambda local, mu visitante) para un enfrentamiento."""
        if local not in self.idx or visit not in self.idx:
            raise KeyError(f"Equipo sin datos en el modelo: {local} / {visit}")
        h, a = self.idx[local], self.idx[visit]
        vent = self.ventaja_de_liga(liga) * (0 if neutral else 1)
        lam = math.exp(self.intercepto + vent + self.ataque[h] - self.defensa[a])
        mu = math.exp(self.intercepto + self.ataque[a] - self.defensa[h])
        return lam, mu

    def matriz_desde_lambdas(self, lam: float, mu: float) -> np.ndarray:
        """Matriz de marcadores a partir de unos goles esperados dados.

        Separada de `matriz_marcador` porque la prórroga de las eliminatorias
        necesita reescalar lambda/mu antes de construir la matriz."""
        from scipy.stats import poisson
        xs = np.arange(self.max_goles + 1)
        M = np.outer(poisson.pmf(xs, lam), poisson.pmf(xs, mu))
        # Corrección DC en los 4 marcadores bajos
        rho = self.rho
        M[0, 0] *= 1.0 - lam * mu * rho
        M[0, 1] *= 1.0 + lam * rho
        M[1, 0] *= 1.0 + mu * rho
        M[1, 1] *= 1.0 - rho
        M = np.clip(M, 0, None)
        return M / M.sum()

    def matriz_marcador(self, local: str, visit: str, neutral: int = 0,
                        liga: str | None = None) -> np.ndarray:
        """Matriz de probabilidad de marcadores M[x, y] = P(local=x, visit=y),
        con la corrección de Dixon-Coles aplicada y normalizada a suma 1."""
        lam, mu = self._lambdas(local, visit, neutral, liga)
        return self.matriz_desde_lambdas(lam, mu)

    def lambdas_publicos(self, local: str, visit: str, neutral: int = 0,
                         liga: str | None = None):
        """Expone (lambda, mu) para mercados como 'primer equipo en anotar'."""
        return self._lambdas(local, visit, neutral, liga)
