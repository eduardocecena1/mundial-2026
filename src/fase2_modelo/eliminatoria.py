"""
eliminatoria.py — Probabilidades de una eliminatoria a doble partido.

Desde febrero la Champions deja de ser "un partido, un pick": lo que importa es
QUIÉN PASA, y eso depende del global de los dos partidos, más la prórroga y los
penales. Este módulo convierte las matrices de marcador del Dixon-Coles en la
distribución del global y de ahí saca la probabilidad de clasificación.

Reglas aplicadas (formato UEFA vigente):
  - No hay regla de goles fuera de casa: la UEFA la abolió en 2021.
  - Global empatado tras la vuelta -> prórroga de 30 minutos en el campo de la
    vuelta. Se modela reescalando lambda y mu a 30/90 de su valor.
  - Si la prórroga tampoco desempata -> penales, 50/50. No hay señal fiable en
    los datos para dar ventaja a nadie en una tanda, así que no la inventamos.

Todo se calcula por convolución exacta sobre la distribución de márgenes, no por
simulación: con matrices de 9x9 es más rápido y no mete ruido de muestreo.
"""

from __future__ import annotations

import numpy as np

# Proporción de la duración de un partido que dura la prórroga (30 de 90 minutos).
FACTOR_PRORROGA = 30.0 / 90.0


def _dist_margen(M: np.ndarray, filas_a_favor: bool = True) -> np.ndarray:
    """Distribución del margen de goles de un equipo, indexada de -K a +K.

    M[x, y] = P(equipo de filas marca x, equipo de columnas marca y).
    Con `filas_a_favor=False` se devuelve el margen del equipo de columnas.
    """
    K = M.shape[0] - 1
    dist = np.zeros(2 * K + 1)
    for x in range(K + 1):
        for y in range(K + 1):
            margen = (x - y) if filas_a_favor else (y - x)
            dist[margen + K] += M[x, y]
    return dist


def _dist_total(M: np.ndarray) -> np.ndarray:
    """Distribución del total de goles del partido (0 .. 2K)."""
    K = M.shape[0] - 1
    dist = np.zeros(2 * K + 1)
    for x in range(K + 1):
        for y in range(K + 1):
            dist[x + y] += M[x, y]
    return dist


def _delta(valor: int, K: int) -> np.ndarray:
    """Distribución degenerada (masa 1 en un valor conocido), centrada en K."""
    dist = np.zeros(2 * K + 1)
    dist[int(np.clip(valor, -K, K)) + K] = 1.0
    return dist


def _prob_desempate(modelo, local_vuelta: str, visitante_vuelta: str,
                    liga: str | None) -> float:
    """P(gana el VISITANTE de la vuelta el desempate: prórroga + penales).

    El visitante de la vuelta es el equipo que jugó la ida en casa. La prórroga
    se juega en el campo de la vuelta, así que la localía sigue siendo del local.
    """
    lam, mu = modelo.lambdas_publicos(local_vuelta, visitante_vuelta, 0, liga)
    M_et = modelo.matriz_desde_lambdas(lam * FACTOR_PRORROGA, mu * FACTOR_PRORROGA)
    K = M_et.shape[0] - 1
    margen = _dist_margen(M_et, filas_a_favor=False)     # a favor del visitante
    p_gana_et = margen[K + 1:].sum()
    p_empate_et = margen[K]
    return float(p_gana_et + 0.5 * p_empate_et)          # penales 50/50


def eliminatoria(modelo, equipo_a: str, equipo_b: str,
                 marcador_ida: tuple[int, int] | None = None,
                 liga: str | None = None) -> dict:
    """Probabilidades de una eliminatoria entre A y B.

    Convención: **A juega la IDA en casa** y B juega la vuelta en casa, que es
    como la UEFA las publica.

    Args:
      marcador_ida: (goles_A, goles_B) si la ida ya se jugó; None si está por jugar.
      liga: liga de la competición, para la ventaja de local correcta.

    Devuelve probabilidades de clasificación, de prórroga, y la distribución del
    global (para mercados de goles de la eliminatoria completa).
    """
    K = modelo.max_goles

    # Ida: A en casa. Margen a favor de A.
    if marcador_ida is None:
        M1 = modelo.matriz_marcador(equipo_a, equipo_b, 0, liga)
        margen1 = _dist_margen(M1, filas_a_favor=True)
        total1 = _dist_total(M1)
    else:
        ga, gb = marcador_ida
        margen1 = _delta(ga - gb, K)
        total1 = np.zeros(2 * K + 1)
        total1[min(ga + gb, 2 * K)] = 1.0

    # Vuelta: B en casa. Margen a favor de A = columnas.
    M2 = modelo.matriz_marcador(equipo_b, equipo_a, 0, liga)
    margen2 = _dist_margen(M2, filas_a_favor=False)
    total2 = _dist_total(M2)

    # Global = suma de los dos márgenes (convolución exacta).
    margen_global = np.convolve(margen1, margen2)        # índice 0 => margen -2K
    centro = 2 * K
    p_a_gana_90 = float(margen_global[centro + 1:].sum())
    p_b_gana_90 = float(margen_global[:centro].sum())
    p_empate_global = float(margen_global[centro])

    # Desempate: prórroga en el campo de B, y penales al 50%.
    p_a_desempate = _prob_desempate(modelo, equipo_b, equipo_a, liga)

    p_pasa_a = p_a_gana_90 + p_empate_global * p_a_desempate
    p_pasa_b = 1.0 - p_pasa_a

    goles_global = np.convolve(total1, total2)

    return {
        "equipo_a": equipo_a,
        "equipo_b": equipo_b,
        "ida_jugada": marcador_ida is not None,
        "marcador_ida": marcador_ida,
        "pasa_a": p_pasa_a,
        "pasa_b": p_pasa_b,
        "gana_a_en_90": p_a_gana_90,
        "gana_b_en_90": p_b_gana_90,
        "prorroga": p_empate_global,
        "penales": p_empate_global * _prob_penales(modelo, equipo_b, equipo_a, liga),
        "goles_global": goles_global,
    }


def _prob_penales(modelo, local_vuelta: str, visitante_vuelta: str,
                  liga: str | None) -> float:
    """P(la eliminatoria llegue a penales | llegó a prórroga)."""
    lam, mu = modelo.lambdas_publicos(local_vuelta, visitante_vuelta, 0, liga)
    M_et = modelo.matriz_desde_lambdas(lam * FACTOR_PRORROGA, mu * FACTOR_PRORROGA)
    K = M_et.shape[0] - 1
    return float(_dist_margen(M_et)[K])


def over_under_global(res: dict, lineas=(1.5, 2.5, 3.5, 4.5, 5.5)) -> dict:
    """Over/Under del total de goles de la eliminatoria completa (los 2 partidos)."""
    goles = res["goles_global"]
    totales = np.arange(len(goles))
    out = {}
    for linea in lineas:
        p_over = float(goles[totales > linea].sum())
        out[f"over_{linea}"] = p_over
        out[f"under_{linea}"] = 1.0 - p_over
    return out


def emparejar_legs(partidos) -> list[dict]:
    """Agrupa filas de `partidos` en eliminatorias de ida y vuelta.

    Empareja por el conjunto de los dos equipos + la ronda. ESPN marca las legs
    con `leg` 1/2, que es lo que decide quién jugó la ida en casa.
    """
    por_llave: dict[tuple, list] = {}
    for p in partidos:
        llave = (p["ronda"], frozenset((p["local_id"], p["visitante_id"])))
        por_llave.setdefault(llave, []).append(p)

    ties = []
    for (ronda, _), legs in por_llave.items():
        legs = sorted(legs, key=lambda r: (r["leg"] or 0, r["fecha"]))
        ida = next((l for l in legs if l["leg"] == 1), legs[0])
        vuelta = next((l for l in legs if l["leg"] == 2), None)
        ties.append({
            "ronda": ronda,
            "equipo_a": ida["local"],          # el que juega la ida en casa
            "equipo_b": ida["visitante"],
            "ida": ida,
            "vuelta": vuelta,
            "marcador_ida": ((ida["goles_local"], ida["goles_visitante"])
                             if ida["jugado"] else None),
        })
    return ties
