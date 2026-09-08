"""
actualizar_diario.py — FASE 1: actualización diaria durante la temporada.

Pensado para correrse cada día de jornada. Refresca la temporada en curso desde
ESPN (UPSERT, sin duplicar), aplica los marcadores en vivo/finales del día y
muestra los partidos de la competición activa para la fecha indicada.

Uso:
    python -m src.fase1_datos.actualizar_diario                 # usa la fecha de hoy
    python -m src.fase1_datos.actualizar_diario --fecha 2026-09-30
    python -m src.fase1_datos.actualizar_diario --sin-descarga  # solo ver el calendario
    python -m src.fase1_datos.actualizar_diario --completo      # re-descarga todo el histórico
"""

import argparse
from datetime import date, timedelta

from ..config import cargar_config, competicion_id, competiciones, nombre_competicion
from . import db
from . import espn_api as espn
from . import marcadores_vivo as mv
from .descargar_clubes import _rango_descarga, derivar_equipos, guardar_partidos


def actualizar(con, cfg: dict, completo: bool = False) -> None:
    """Refresca la base desde ESPN (por defecto solo la temporada en curso)."""
    desde, hasta = _rango_descarga(cfg, solo_actual=not completo)
    slugs = list(competiciones(cfg).keys())
    print(f"Actualizando {len(slugs)} competiciones ({desde} -> {hasta})...")

    total = 0
    for slug in slugs:
        try:
            partidos = espn.partidos_rango(slug, desde, hasta, usar_cache=not completo)
        except espn.ESPNNoDisponible:
            continue                     # esa competición no está en ESPN; ya avisamos al ingerir
        total += guardar_partidos(con, partidos)
    derivar_equipos(con, cfg)
    print(f"  {total} partidos refrescados.")

    # Marcadores en vivo/finales de hoy y ayer (una jornada puede cruzar medianoche UTC).
    hoy = date.today()
    for d in (hoy, hoy - timedelta(days=1)):
        try:
            res = mv.aplicar(con, d.isoformat(), cfg)
            if res["finales"] or res["goles"]:
                print(f"  ESPN en vivo ({d}): {res['finales']} finales, "
                      f"{res['goles']} goles guardados")
        except Exception:
            pass                          # si ESPN falla, seguimos con lo ya descargado

    db.set_meta(con, "ultima_actualizacion", hoy.isoformat())


def mostrar_dia(con, fecha: str, cfg: dict) -> None:
    """Imprime el calendario de la competición activa para una fecha."""
    partidos = db.calendario_de_fecha(con, fecha)
    titulo = nombre_competicion(cfg).upper()
    print(f"\n=== PARTIDOS — {titulo} — {fecha} ===")
    if not partidos:
        print("  (No hay partidos programados para esta fecha.)")
        return
    for p in partidos:
        if p["jugado"]:
            estado = f"FINAL  {p['goles_local']}-{p['goles_visitante']}"
        else:
            estado = f"por jugar {mv.hora_mexico(p['fecha_hora'])} MX"
        sede = p["ciudad"] or "?"
        print(f"  {p['local']:26s} vs {p['visitante']:26s} [{estado}]  ({sede})")


def main():
    parser = argparse.ArgumentParser(description="Actualización diaria de datos.")
    parser.add_argument("--fecha", default=date.today().isoformat(),
                        help="Fecha a mostrar YYYY-MM-DD (defecto: hoy).")
    parser.add_argument("--sin-descarga", action="store_true",
                        help="No re-descargar; solo mostrar el calendario de la fecha.")
    parser.add_argument("--completo", action="store_true",
                        help="Re-descargar todo el histórico ignorando la caché.")
    args = parser.parse_args()

    cfg = cargar_config()
    con = db.conectar()
    db.inicializar(con)

    if not args.sin_descarga:
        actualizar(con, cfg, completo=args.completo)
        comp = competicion_id(cfg)
        fila = con.execute(
            "SELECT MAX(temporada) t FROM partidos WHERE competicion=?", (comp,)).fetchone()
        temporada = fila["t"]
        totales = con.execute(
            "SELECT COUNT(*) n, SUM(jugado) j FROM partidos "
            "WHERE competicion=? AND temporada=?", (comp, temporada)).fetchone()
        print(f"Base actualizada. {nombre_competicion(cfg)}: "
              f"{totales['j'] or 0}/{totales['n']} partidos jugados.")

    mostrar_dia(con, args.fecha, cfg)
    con.close()


if __name__ == "__main__":
    main()
