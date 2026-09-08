"""
simulacion.py — Monte Carlo de la fase liga (formato de 36 equipos).

Desde 2024-25 la Champions no tiene grupos: los 36 equipos comparten una única
tabla y cada uno juega 8 partidos contra rivales distintos. Al final:

    posiciones  1-8   -> octavos directos
    posiciones  9-24  -> playoff de acceso a octavos (ida y vuelta)
    posiciones 25-36  -> eliminados

Este módulo simula los partidos que faltan muestreando de la matriz de marcador
del Dixon-Coles (los ya jugados entran con su resultado real) y devuelve, por
club, la probabilidad de acabar en cada tramo. Es la foto de contexto que un
pick suelto no da: un partido puede importar poco o muchísimo según dónde deje
al equipo en la tabla.

Desempates: puntos -> diferencia de goles -> goles a favor -> aleatorio. La UEFA
usa después criterios que no modelamos (victorias, disciplina, coeficiente), así
que el desempate final es aleatorio a propósito en vez de fingir precisión.
"""

from __future__ import annotations

import numpy as np

PUNTOS_VICTORIA = 3
PUNTOS_EMPATE = 1


def _muestrear_marcadores(M: np.ndarray, n_sims: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Muestrea n_sims marcadores (goles_local, goles_visitante) de la matriz M."""
    K = M.shape[0]
    plano = M.ravel() / M.sum()
    idx = rng.choice(K * K, size=n_sims, p=plano)
    return idx // K, idx % K


def simular_fase_liga(modelo, partidos, cfg: dict, liga: str | None = "uefa",
                      n_sims: int | None = None, semilla: int = 20262027) -> dict:
    """Simula la tabla de la fase liga.

    Args:
      partidos: filas de la tabla `partidos` de la fase liga (jugados y pendientes).
      liga: liga de la competición, para la ventaja de local del modelo.

    Devuelve {"equipos": {nombre: {...}}, "n_sims", "jugados", "pendientes"}.
    """
    sim_cfg = cfg.get("simulacion", {})
    n_sims = n_sims or sim_cfg.get("n_sims", 10000)
    plazas_directas = sim_cfg.get("plazas_directas", 8)
    plazas_playoff = sim_cfg.get("plazas_playoff", 24)
    rng = np.random.default_rng(semilla)

    equipos = sorted({p["local"] for p in partidos} | {p["visitante"] for p in partidos})
    idx = {e: i for i, e in enumerate(equipos)}
    n_eq = len(equipos)

    puntos = np.zeros((n_sims, n_eq), dtype=np.int32)
    gf = np.zeros((n_sims, n_eq), dtype=np.int32)
    gc = np.zeros((n_sims, n_eq), dtype=np.int32)

    n_jugados = n_pendientes = 0

    for p in partidos:
        i, j = idx[p["local"]], idx[p["visitante"]]

        if p["jugado"] and p["goles_local"] is not None:
            gl = np.full(n_sims, p["goles_local"], dtype=np.int32)
            gv = np.full(n_sims, p["goles_visitante"], dtype=np.int32)
            n_jugados += 1
        else:
            try:
                M = modelo.matriz_marcador(p["local"], p["visitante"], p["neutral"], liga)
            except KeyError:
                continue                 # equipo sin datos: ese partido no se simula
            gl, gv = _muestrear_marcadores(M, n_sims, rng)
            gl = gl.astype(np.int32)
            gv = gv.astype(np.int32)
            n_pendientes += 1

        gf[:, i] += gl; gc[:, i] += gv
        gf[:, j] += gv; gc[:, j] += gl

        local_gana = gl > gv
        empate = gl == gv
        puntos[:, i] += np.where(local_gana, PUNTOS_VICTORIA,
                                 np.where(empate, PUNTOS_EMPATE, 0))
        puntos[:, j] += np.where(gl < gv, PUNTOS_VICTORIA,
                                 np.where(empate, PUNTOS_EMPATE, 0))

    # Ordenar cada simulación: puntos -> diferencia -> goles a favor -> azar.
    dif = gf - gc
    desempate = rng.random((n_sims, n_eq))
    # np.lexsort ordena ascendente por la ÚLTIMA clave, así que van invertidas.
    orden = np.lexsort((desempate, gf, dif, puntos), axis=1)[:, ::-1]

    # posicion[s, e] = puesto (1..n) del equipo e en la simulación s
    posicion = np.empty((n_sims, n_eq), dtype=np.int32)
    filas = np.arange(n_sims)[:, None]
    posicion[filas, orden] = np.arange(1, n_eq + 1)[None, :]

    resultado = {}
    for e, i in idx.items():
        pos = posicion[:, i]
        resultado[e] = {
            "top_directo": float(np.mean(pos <= plazas_directas)),
            "playoff": float(np.mean((pos > plazas_directas) & (pos <= plazas_playoff))),
            "eliminado": float(np.mean(pos > plazas_playoff)),
            "posicion_media": float(np.mean(pos)),
            "puntos_medios": float(np.mean(puntos[:, i])),
        }

    return {"equipos": resultado, "n_sims": n_sims,
            "jugados": n_jugados, "pendientes": n_pendientes,
            "plazas_directas": plazas_directas, "plazas_playoff": plazas_playoff}


def partidos_fase_liga(con, comp: str, temporada: int, ronda: str = "league-phase"):
    """Todos los partidos de la fase liga de una temporada."""
    return con.execute(
        """SELECT local, visitante, goles_local, goles_visitante, neutral, jugado
             FROM partidos
            WHERE competicion=? AND temporada=? AND ronda=?
            ORDER BY fecha""",
        (comp, temporada, ronda),
    ).fetchall()


def tabla_ordenada(sim: dict) -> list[tuple[str, dict]]:
    """Equipos ordenados por probabilidad de acabar en el top directo."""
    return sorted(sim["equipos"].items(),
                  key=lambda kv: (-kv[1]["top_directo"], kv[1]["posicion_media"]))
