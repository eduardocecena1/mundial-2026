"""
contexto.py — Datos de contexto para justificar las recomendaciones.

Calcula la forma reciente de cada selección y el head-to-head directo, para que
cada apuesta recomendada venga con una explicación "basada en datos" (forma,
promedio de goles, historial directo), como pide el proyecto.
"""

from __future__ import annotations

from datetime import date


def forma_reciente(con, equipo: str, hasta: str, n: int = 8) -> dict:
    """Resumen de los últimos 'n' partidos jugados del equipo antes de 'hasta'.

    Devuelve victorias/empates/derrotas y promedios de goles a favor/en contra.
    """
    filas = con.execute(
        """SELECT local, visitante, goles_local, goles_visitante
             FROM partidos
            WHERE jugado=1 AND fecha < ? AND (local=? OR visitante=?)
            ORDER BY fecha DESC LIMIT ?""",
        (hasta, equipo, equipo, n),
    ).fetchall()

    v = e = d = 0
    gf = gc = 0
    for f in filas:
        es_local = f["local"] == equipo
        propios = f["goles_local"] if es_local else f["goles_visitante"]
        rival = f["goles_visitante"] if es_local else f["goles_local"]
        gf += propios
        gc += rival
        if propios > rival:
            v += 1
        elif propios == rival:
            e += 1
        else:
            d += 1

    k = len(filas)
    return {
        "n": k,
        "v": v, "e": e, "d": d,
        "gf_prom": (gf / k) if k else 0.0,
        "gc_prom": (gc / k) if k else 0.0,
        "racha": "".join(_letra(f, equipo) for f in filas[:5]),  # ej. 'WWDLW' (reciente->antiguo)
    }


def _letra(fila, equipo: str) -> str:
    es_local = fila["local"] == equipo
    propios = fila["goles_local"] if es_local else fila["goles_visitante"]
    rival = fila["goles_visitante"] if es_local else fila["goles_local"]
    return "V" if propios > rival else ("E" if propios == rival else "D")


def head_to_head(con, a: str, b: str, hasta: str, anios: int = 12) -> dict:
    """Historial directo entre dos equipos en los últimos 'anios'."""
    desde = date.fromisoformat(hasta).replace(
        year=date.fromisoformat(hasta).year - anios).isoformat()
    filas = con.execute(
        """SELECT local, visitante, goles_local, goles_visitante
             FROM partidos
            WHERE jugado=1 AND fecha < ? AND fecha >= ?
              AND ((local=? AND visitante=?) OR (local=? AND visitante=?))
            ORDER BY fecha DESC""",
        (hasta, desde, a, b, b, a),
    ).fetchall()

    gana_a = gana_b = empates = 0
    for f in filas:
        ga = f["goles_local"] if f["local"] == a else f["goles_visitante"]
        gb = f["goles_local"] if f["local"] == b else f["goles_visitante"]
        if ga > gb:
            gana_a += 1
        elif ga < gb:
            gana_b += 1
        else:
            empates += 1
    return {"n": len(filas), "gana_a": gana_a, "gana_b": gana_b, "empates": empates}
