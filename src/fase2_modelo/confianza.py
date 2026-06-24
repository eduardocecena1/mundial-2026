"""
confianza.py — Nivel de confianza de cada predicción.

La idea (requisito del proyecto): un equipo con muchos partidos históricos da
estimaciones fiables; uno con pocos datos, no. En vez de inventar números,
bajamos la confianza. El nivel se basa en el MÍNIMO de partidos jugados entre
los dos equipos del enfrentamiento (la cadena es tan fuerte como su eslabón
más débil) y, además, distingue mercados sólidos (goles) de derivados (tarjetas).
"""

from __future__ import annotations


def nivel_por_datos(n_min: int, umbral_alto: int = 40, umbral_medio: int = 15) -> str:
    """Devuelve 'alto' | 'medio' | 'bajo' según el nº de partidos disponibles."""
    if n_min >= umbral_alto:
        return "alto"
    if n_min >= umbral_medio:
        return "medio"
    return "bajo"


def confianza_partido(con, local: str, visit: str, cfg: dict) -> dict:
    """Calcula el nivel de confianza para un enfrentamiento dado.

    Devuelve un dict con el nº de partidos de cada equipo, el mínimo y el nivel.
    """
    from ..fase1_datos import db
    desde = cfg.get("datos", {}).get("desde", "2014-01-01")
    umbrales = cfg.get("confianza", {})
    n_local = db.num_partidos_equipo(con, local, desde)
    n_visit = db.num_partidos_equipo(con, visit, desde)
    n_min = min(n_local, n_visit)
    nivel = nivel_por_datos(
        n_min,
        umbrales.get("alto", 40),
        umbrales.get("medio", 15),
    )
    return {
        "n_local": n_local,
        "n_visit": n_visit,
        "n_min": n_min,
        "nivel": nivel,
    }
