"""
actualizar_diario.py — FASE 1: actualización diaria durante el Mundial.

Pensado para correrse cada día del torneo. Vuelve a descargar el dataset (que se
actualiza a diario en origen), refresca la base con UPSERT —rellenando los
marcadores de los partidos que ya se jugaron— y muestra:
  - qué resultados nuevos entraron,
  - los partidos del Mundial 2026 programados para la fecha indicada.

Uso:
    python -m src.fase1_datos.actualizar_diario                # usa la fecha de hoy
    python -m src.fase1_datos.actualizar_diario --fecha 2026-06-25
"""

import argparse
from datetime import date

from . import db
from .descargar_historico import (
    ARCHIVOS, _descargar_csv, cargar_results, cargar_goleadores,
    cargar_shootouts, derivar_equipos, DESDE_DEFECTO,
)


def actualizar(con) -> None:
    """Re-descarga las fuentes y refresca la base (marcadores nuevos incluidos)."""
    print("Actualizando datos desde el origen...")
    df_results = _descargar_csv(ARCHIVOS["results"])
    df_goles = _descargar_csv(ARCHIVOS["goalscorers"])
    df_shoot = _descargar_csv(ARCHIVOS["shootouts"])

    cargar_results(con, df_results, DESDE_DEFECTO)
    cargar_goleadores(con, df_goles, DESDE_DEFECTO)
    cargar_shootouts(con, df_shoot, DESDE_DEFECTO)
    derivar_equipos(con)
    db.set_meta(con, "ultima_actualizacion", date.today().isoformat())


def mostrar_dia(con, fecha: str) -> None:
    """Imprime el calendario del Mundial 2026 para una fecha y el estado de cada partido."""
    partidos = db.calendario_de_fecha(con, fecha)
    print(f"\n=== PARTIDOS DEL MUNDIAL 2026 — {fecha} ===")
    if not partidos:
        print("  (No hay partidos del Mundial 2026 programados para esta fecha.)")
        return
    for p in partidos:
        if p["jugado"]:
            estado = f"FINAL  {p['goles_local']}-{p['goles_visitante']}"
        else:
            estado = "por jugar"
        print(f"  {p['local']} vs {p['visitante']:24s} [{estado}]  ({p['ciudad']})")


def main():
    parser = argparse.ArgumentParser(description="Actualización diaria de datos.")
    parser.add_argument("--fecha", default=date.today().isoformat(),
                        help="Fecha a mostrar YYYY-MM-DD (defecto: hoy).")
    parser.add_argument("--sin-descarga", action="store_true",
                        help="No re-descargar; solo mostrar el calendario de la fecha.")
    args = parser.parse_args()

    con = db.conectar()
    db.inicializar(con)

    if not args.sin_descarga:
        actualizar(con)
        jugados = con.execute(
            "SELECT COUNT(*) n FROM partidos WHERE es_mundial2026=1 AND jugado=1"
        ).fetchone()["n"]
        print(f"Base actualizada. Partidos del Mundial 2026 ya jugados: {jugados}/72")

    mostrar_dia(con, args.fecha)
    con.close()


if __name__ == "__main__":
    main()
