"""
poisson_dixon_coles.py — FASE 2: motor principal de predicción de goles.

Implementa el modelo de Dixon-Coles (1997), una extensión del modelo de Poisson
bivariado pensada justamente para fútbol:

  goles_local    ~ Poisson(lambda),  log(lambda) = c + ventaja_local*(no_neutral)
                                                    + ataque[local] - defensa[visit]
  goles_visit    ~ Poisson(mu),      log(mu)     = c + ataque[visit] - defensa[local]

Mejoras sobre el Poisson simple, todas orientadas a PRECISIÓN:

  1. Corrección de Dixon-Coles (parámetro rho) para los marcadores bajos
     (0-0, 1-0, 0-1, 1-1), que el Poisson puro estima mal.
  2. Verosimilitud PONDERADA:
       - decaimiento temporal: los partidos recientes pesan más (vida media配config).
       - peso por torneo: un Mundial informa más que un amistoso.
  3. Campo neutral: la ventaja de localía solo se aplica si NO es campo neutral
     (clave en datos de selecciones, donde muchos partidos son en sede neutral).
  4. Regularización L2 (ridge): encoge ataque/defensa hacia 0 (la media) para los
     equipos con pocos partidos -> evita estimaciones disparatadas y conecta con
     el nivel de confianza. Implementa el requisito de "no inventar números".

Se entrena por máxima verosimilitud con gradiente ANALÍTICO (L-BFGS-B), lo que
permite ajustar cientos de selecciones en segundos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


# --- Pesos (decaimiento temporal + torneo) ----------------------------------

def peso_torneo(torneo: str, pesos: dict) -> float:
    """Peso de importancia de un torneo (amistosos informan menos que Mundiales)."""
    return float(pesos.get(torneo, pesos.get("_defecto", 0.7)))


def peso_temporal(edad_anios: np.ndarray, vida_media_anios: float) -> np.ndarray:
    """Decaimiento exponencial: w = 0.5 ** (edad / vida_media)."""
    xi = math.log(2.0) / max(vida_media_anios, 1e-6)
    return np.exp(-xi * np.clip(edad_anios, 0.0, None))


# --- El modelo --------------------------------------------------------------

@dataclass
class DixonColes:
    max_goles: int = 8
    reg: float = 1.0                  # fuerza de la regularización L2 (ridge)
    # rellenados tras fit():
    equipos: list = field(default_factory=list)
    idx: dict = field(default_factory=dict)
    ataque: np.ndarray = None
    defensa: np.ndarray = None
    intercepto: float = 0.0
    ventaja_local: float = 0.0
    rho: float = 0.0
    _ajustado: bool = False

    # ---- Entrenamiento ----

    def fit(self, local, visit, gl, gv, neutral, peso):
        """Ajusta los parámetros por máxima verosimilitud ponderada.

        Args (todos arrays de igual longitud, una entrada por partido jugado):
          local, visit : nombres de equipo (local y visitante)
          gl, gv       : goles de local y visitante (int)
          neutral      : 1 si campo neutral, 0 si hay localía real
          peso         : peso del partido (decaimiento temporal * peso torneo)
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

        # Máscaras de los marcadores bajos para la corrección de Dixon-Coles
        m00 = (gl == 0) & (gv == 0)
        m01 = (gl == 0) & (gv == 1)
        m10 = (gl == 1) & (gv == 0)
        m11 = (gl == 1) & (gv == 1)
        no_neutral = 1.0 - neutral

        # Vector de parámetros: [ataque(n), defensa(n), intercepto, ventaja, rho]
        def desempaqueta(p):
            atk = p[:n]
            dfn = p[n:2 * n]
            c, g, rho = p[2 * n], p[2 * n + 1], p[2 * n + 2]
            return atk, dfn, c, g, rho

        def nll_y_grad(p):
            atk, dfn, c, g, rho = desempaqueta(p)
            eta1 = c + g * no_neutral + atk[ih] - dfn[ia]
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

            nll = -np.sum(w * ll) + self.reg * (np.sum(atk ** 2) + np.sum(dfn ** 2))

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
            grad = np.zeros_like(p)
            # ataque/defensa (acumulación por equipo)
            np.add.at(grad, ih, -g_eta1)                 # d eta1/d atk[local]=+1
            np.add.at(grad, ia, -g_eta2)                 # d eta2/d atk[visit]=+1
            np.add.at(grad, n + ia, +g_eta1)             # d eta1/d def[visit]=-1
            np.add.at(grad, n + ih, +g_eta2)             # d eta2/d def[local]=-1
            grad[2 * n] = -np.sum(g_eta1 + g_eta2)       # intercepto
            grad[2 * n + 1] = -np.sum(g_eta1 * no_neutral)  # ventaja local
            grad[2 * n + 2] = -g_rho                     # rho
            # término de regularización
            grad[:n] += 2 * self.reg * atk
            grad[n:2 * n] += 2 * self.reg * dfn
            return nll, grad

        # Inicialización e límites
        p0 = np.zeros(2 * n + 3)
        p0[2 * n] = math.log(max(np.average(gl + gv, weights=w) / 2.0, 0.1))  # intercepto
        p0[2 * n + 1] = 0.25                                                   # ventaja local
        p0[2 * n + 2] = -0.05                                                  # rho
        limites = [(-3, 3)] * (2 * n) + [(-1, 2), (-0.5, 1.0), (-0.2, 0.2)]

        res = minimize(nll_y_grad, p0, jac=True, method="L-BFGS-B",
                       bounds=limites, options={"maxiter": 500, "ftol": 1e-9})

        atk, dfn, c, g, rho = desempaqueta(res.x)
        # Recentrar ataque/defensa a media 0 (identificabilidad; no cambia predicciones)
        atk = atk - atk.mean()
        dfn = dfn - dfn.mean()
        self.ataque, self.defensa = atk, dfn
        self.intercepto, self.ventaja_local, self.rho = c, g, rho
        self._ajustado = True
        return self

    # ---- Predicción ----

    def _lambdas(self, local: str, visit: str, neutral: int):
        """Goles esperados (lambda local, mu visitante) para un enfrentamiento."""
        if local not in self.idx or visit not in self.idx:
            raise KeyError(f"Equipo sin datos en el modelo: {local} / {visit}")
        h, a = self.idx[local], self.idx[visit]
        vent = self.ventaja_local * (0 if neutral else 1)
        lam = math.exp(self.intercepto + vent + self.ataque[h] - self.defensa[a])
        mu = math.exp(self.intercepto + self.ataque[a] - self.defensa[h])
        return lam, mu

    def matriz_marcador(self, local: str, visit: str, neutral: int = 1) -> np.ndarray:
        """Matriz de probabilidad de marcadores M[x, y] = P(local=x, visit=y),
        con la corrección de Dixon-Coles aplicada y normalizada a suma 1."""
        lam, mu = self._lambdas(local, visit, neutral)
        K = self.max_goles
        xs = np.arange(K + 1)
        # Poisson independiente
        from scipy.stats import poisson
        ph = poisson.pmf(xs, lam)
        pa = poisson.pmf(xs, mu)
        M = np.outer(ph, pa)
        # Corrección DC en los 4 marcadores bajos
        rho = self.rho
        M[0, 0] *= 1.0 - lam * mu * rho
        M[0, 1] *= 1.0 + lam * rho
        M[1, 0] *= 1.0 + mu * rho
        M[1, 1] *= 1.0 - rho
        M = np.clip(M, 0, None)
        M /= M.sum()
        return M

    def lambdas_publicos(self, local: str, visit: str, neutral: int = 1):
        """Expone (lambda, mu) para mercados como 'primer equipo en anotar'."""
        return self._lambdas(local, visit, neutral)
