"""
predecir_partido.py — Orquesta la predicción completa de un partido.

Junta el modelo de goles (Dixon-Coles), todos los mercados derivados y el nivel
de confianza en un único objeto de predicción, listo para que la Fase 3
(recomendación) y la Fase 4 (interfaz) lo consuman.
"""

from __future__ import annotations

from ..config import liga_de_competicion
from . import mercados
from .confianza import confianza_partido


def predecir(con, modelo, cfg: dict, local: str, visit: str, neutral: int = 0,
             competicion: str | None = None) -> dict:
    """Predicción completa de un enfrentamiento.

    Devuelve un dict con todos los mercados (probabilidades 0..1), los goles
    esperados y el nivel de confianza basado en datos disponibles.

    `neutral` es 0 por defecto: en fútbol de clubes casi todo se juega con
    localía real (solo la final de Champions es campo neutral). En el modelo
    del Mundial el defecto era el contrario.
    """
    lineas = tuple(cfg["apuestas"]["lineas_goles"])
    liga = liga_de_competicion(competicion, cfg) if competicion else None
    merc = mercados.todos_los_mercados(modelo, local, visit, neutral, lineas, liga)
    conf = confianza_partido(con, local, visit, cfg)
    return {
        "local": local,
        "visitante": visit,
        "neutral": bool(neutral),
        "competicion": competicion,
        "confianza": conf,
        **merc,
    }


def _pct(x: float) -> str:
    return f"{100 * x:4.1f}%"


def imprimir_prediccion(p: dict) -> None:
    """Imprime una predicción de forma legible (para depurar / CLI simple)."""
    loc, vis = p["local"], p["visitante"]
    sede = "neutral" if p["neutral"] else "con localía"
    c = p["confianza"]
    print(f"\n{'='*64}")
    print(f"  {loc}  vs  {vis}   ({sede})")
    print(f"  Goles esperados: {loc} {p['lambda_local']:.2f} - "
          f"{p['lambda_visit']:.2f} {vis}")
    print(f"  Confianza: {c['nivel'].upper()}  "
          f"(datos: {loc} {c['n_local']}, {vis} {c['n_visit']} partidos)")
    print(f"{'-'*64}")
    x = p["1x2"]
    print(f"  1X2     Local {_pct(x['local'])} | Empate {_pct(x['empate'])} "
          f"| Visit {_pct(x['visitante'])}")
    d = p["doble_oportunidad"]
    print(f"  Doble   1X {_pct(d['1X'])} | 12 {_pct(d['12'])} | X2 {_pct(d['X2'])}")
    ou = p["over_under"]
    print(f"  O/U 2.5 Over {_pct(ou['over_2.5'])} | Under {_pct(ou['under_2.5'])}")
    b = p["btts"]
    print(f"  BTTS    Sí {_pct(b['si'])} | No {_pct(b['no'])}")
    print("  Marcadores más probables: " + ", ".join(
        f"{i}-{j} ({_pct(pr)})" for i, j, pr in p["marcadores_top3"]))
    fa = p["primer_en_anotar"]
    print(f"  1er gol  {loc} {_pct(fa['local'])} | {vis} {_pct(fa['visitante'])} "
          f"| sin goles {_pct(fa['sin_goles'])}")
    h = p["handicap_sugerido"]
    print(f"  Hándicap {h['favorito']} {h['linea']:+.1f} -> "
          f"cubre {_pct(h['local_cubre'] if h['favorito']=='local' else h['visitante_cubre'])}")
