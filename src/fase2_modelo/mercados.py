"""
mercados.py — Deriva todos los mercados de apuestas a partir del modelo.

Todo sale de UNA matriz de marcadores coherente (M[x, y] = P(local x, visit y)),
por lo que los mercados nunca se contradicen entre sí. Cada función devuelve
probabilidades (0..1).
"""

from __future__ import annotations

import math

import numpy as np


def resultado_1x2(M: np.ndarray) -> dict:
    """Gana local / Empate / Gana visitante."""
    x = np.arange(M.shape[0])[:, None]
    y = np.arange(M.shape[1])[None, :]
    return {
        "local": float(M[x > y].sum()),
        "empate": float(M[x == y].sum()),
        "visitante": float(M[x < y].sum()),
    }


def doble_oportunidad(p1x2: dict) -> dict:
    """1X (local o empate), 12 (no empate), X2 (empate o visitante)."""
    return {
        "1X": p1x2["local"] + p1x2["empate"],
        "12": p1x2["local"] + p1x2["visitante"],
        "X2": p1x2["empate"] + p1x2["visitante"],
    }


def over_under(M: np.ndarray, lineas=(1.5, 2.5, 3.5)) -> dict:
    """Over/Under de goles totales para cada línea."""
    x = np.arange(M.shape[0])[:, None]
    y = np.arange(M.shape[1])[None, :]
    total = x + y
    out = {}
    for L in lineas:
        over = float(M[total > L].sum())
        out[f"over_{L}"] = over
        out[f"under_{L}"] = 1.0 - over
    return out


def btts(M: np.ndarray) -> dict:
    """Ambos equipos anotan (Both Teams To Score): Sí / No."""
    si = float(M[1:, 1:].sum())
    return {"si": si, "no": 1.0 - si}


def marcadores_top(M: np.ndarray, top: int = 3) -> list:
    """Top-N marcadores exactos más probables: [(x, y, prob), ...]."""
    idx = np.dstack(np.unravel_index(np.argsort(M.ravel())[::-1], M.shape))[0]
    return [(int(i), int(j), float(M[i, j])) for i, j in idx[:top]]


def primer_en_anotar(lam: float, mu: float) -> dict:
    """Primer equipo en anotar, modelando los goles como dos procesos de Poisson
    independientes con tasas lambda y mu durante el partido:
        P(local primero) = lambda/(lambda+mu) * P(hay al menos un gol)
        P(sin goles)     = exp(-(lambda+mu))
    """
    s = lam + mu
    p_algun_gol = 1.0 - math.exp(-s)
    if s == 0:
        return {"local": 0.0, "visitante": 0.0, "sin_goles": 1.0}
    return {
        "local": (lam / s) * p_algun_gol,
        "visitante": (mu / s) * p_algun_gol,
        "sin_goles": math.exp(-s),
    }


def handicap_asiatico(M: np.ndarray, handicap: float) -> dict:
    """Hándicap asiático simple para el LOCAL con la línea dada (ej. -1.5, +1.0).
    El local 'cubre' si (goles_local - goles_visit + handicap) > 0.
    Para líneas enteras existe el 'push' (devolución) cuando la diferencia es 0;
    aquí lo repartimos como no-cubre para mantener una probabilidad simple."""
    x = np.arange(M.shape[0])[:, None]
    y = np.arange(M.shape[1])[None, :]
    margen = (x - y) + handicap
    cubre = float(M[margen > 0].sum())
    return {"local_cubre": cubre, "visitante_cubre": 1.0 - cubre}


def sugerir_handicap(p1x2: dict, lam: float, mu: float) -> dict:
    """Elige un hándicap informativo según la diferencia de nivel.
    Si un equipo es claro favorito (>60%), propone -1.5 a su favor; si no,
    propone la línea 0 (equivalente a 'empate no cuenta')."""
    fav_local = p1x2["local"] >= p1x2["visitante"]
    diff = abs(lam - mu)
    if max(p1x2["local"], p1x2["visitante"]) > 0.60 and diff > 0.8:
        linea = -1.5 if fav_local else 1.5
    else:
        linea = -0.5 if fav_local else 0.5
    return {"favorito": "local" if fav_local else "visitante", "linea": linea}


def todos_los_mercados(modelo, local: str, visit: str, neutral: int,
                       lineas_goles=(1.5, 2.5, 3.5)) -> dict:
    """Calcula todos los mercados de goles de un partido en un solo dict."""
    M = modelo.matriz_marcador(local, visit, neutral)
    lam, mu = modelo.lambdas_publicos(local, visit, neutral)
    p1x2 = resultado_1x2(M)
    hcap = sugerir_handicap(p1x2, lam, mu)
    return {
        "lambda_local": lam,
        "lambda_visit": mu,
        "1x2": p1x2,
        "doble_oportunidad": doble_oportunidad(p1x2),
        "over_under": over_under(M, lineas_goles),
        "btts": btts(M),
        "marcadores_top3": marcadores_top(M, 3),
        "primer_en_anotar": primer_en_anotar(lam, mu),
        "handicap_sugerido": {**hcap, **handicap_asiatico(M, hcap["linea"])},
    }
