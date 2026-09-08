"""
descargar_clubes.py — Ingesta del histórico de clubes desde ESPN a SQLite.

Recorre todas las competiciones de `config.yaml` (`datos.competiciones`) desde
`datos.desde` hasta el final del calendario ya publicado, y hace UPSERT en
`partidos`. Después deriva la tabla `equipos` con la liga de cada club, que es
lo que necesita el modelo jerárquico.

Uso:
    python -m src.fase1_datos.descargar_clubes                  # todo el histórico
    python -m src.fase1_datos.descargar_clubes --desde 2018-07-01
    python -m src.fase1_datos.descargar_clubes --solo-actual    # solo temporada en curso
    python -m src.fase1_datos.descargar_clubes --competicion eng.1

La primera pasada tarda 10-20 minutos y dejará la respuesta cruda cacheada en
`data/raw/espn/`; las siguientes solo bajan lo que ha cambiado.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date, datetime, timedelta

from ..config import cargar_config, competicion_id, competiciones, ruta_db
from . import db
from . import espn_api as espn

# Cuánto calendario futuro pedimos: cubre hasta la final de la temporada en curso.
DIAS_FUTURO = 400


def _rango_descarga(cfg: dict, solo_actual: bool) -> tuple[date, date]:
    hoy = datetime.now().date()
    if solo_actual:
        # Temporada europea: arranca el 1 de julio.
        inicio_temporada = date(hoy.year if hoy.month >= 7 else hoy.year - 1, 7, 1)
        return inicio_temporada, hoy + timedelta(days=DIAS_FUTURO)
    desde = datetime.strptime(cfg["datos"]["desde"], "%Y-%m-%d").date()
    return desde, hoy + timedelta(days=DIAS_FUTURO)


def guardar_partidos(con: sqlite3.Connection, partidos: list[dict]) -> int:
    """UPSERT por `espn_id`. Devuelve cuántas filas se insertaron o actualizaron.

    El UPSERT es lo que permite volver a correr la descarga sin duplicar: un
    partido que estaba programado y ya se jugó simplemente gana su marcador.
    """
    if not partidos:
        return 0

    # El dict de espn_api trae campos extra (marcador en vivo, global de la
    # eliminatoria) que no son columnas de `partidos`.
    columnas = ("espn_id", "fecha", "fecha_hora", "local", "visitante", "local_id",
                "visitante_id", "goles_local", "goles_visitante", "competicion",
                "temporada", "ronda", "leg", "ciudad", "pais", "neutral", "jugado")
    partidos = [{c: p.get(c) for c in columnas} for p in partidos]

    con.executemany(
        """INSERT INTO partidos
             (espn_id, fecha, fecha_hora, local, visitante, local_id, visitante_id,
              goles_local, goles_visitante, competicion, temporada, ronda, leg,
              ciudad, pais, neutral, jugado)
           VALUES (:espn_id, :fecha, :fecha_hora, :local, :visitante, :local_id,
                   :visitante_id, :goles_local, :goles_visitante, :competicion,
                   :temporada, :ronda, :leg, :ciudad, :pais, :neutral, :jugado)
           ON CONFLICT(espn_id) DO UPDATE SET
             fecha           = excluded.fecha,
             fecha_hora      = excluded.fecha_hora,
             local           = excluded.local,
             visitante       = excluded.visitante,
             local_id        = excluded.local_id,
             visitante_id    = excluded.visitante_id,
             goles_local     = excluded.goles_local,
             goles_visitante = excluded.goles_visitante,
             temporada       = excluded.temporada,
             ronda           = excluded.ronda,
             leg             = excluded.leg,
             ciudad          = excluded.ciudad,
             pais            = excluded.pais,
             neutral         = excluded.neutral,
             jugado          = excluded.jugado""",
        partidos,
    )
    con.commit()
    return len(partidos)


def derivar_equipos(con: sqlite3.Connection, cfg: dict | None = None) -> int:
    """Rellena `equipos` a partir de los partidos ya descargados.

    La liga de un club es la de la competición doméstica más reciente en la que
    jugó. Los clubes que solo aparecen en competiciones UEFA (porque ESPN no
    cubre su liga: Chequia, Ucrania, Eslovaquia, Azerbaiyán) caen en 'otras' y
    el modelo jerárquico los tratará como un grupo aparte.
    """
    cfg = cfg or cargar_config()
    ligas = {slug: meta.get("liga", "otras")
             for slug, meta in competiciones(cfg).items()}
    objetivo = competicion_id(cfg)

    # Un equipo, una fila por cada partido suyo: (id, nombre, competicion, temporada)
    filas = con.execute(
        """SELECT local_id AS id, local AS nombre, competicion, temporada FROM partidos
           WHERE local_id IS NOT NULL
           UNION ALL
           SELECT visitante_id AS id, visitante AS nombre, competicion, temporada FROM partidos
           WHERE visitante_id IS NOT NULL"""
    ).fetchall()

    nombres: dict[str, str] = {}
    mejor_domestica: dict[str, tuple[int, str]] = {}   # id -> (temporada, liga)
    for f in filas:
        eid = f["id"]
        nombres[eid] = f["nombre"]                     # se queda el nombre más reciente visto
        liga = ligas.get(f["competicion"], "otras")
        if liga == "uefa":
            continue
        temporada = f["temporada"] or 0
        if eid not in mejor_domestica or temporada > mejor_domestica[eid][0]:
            mejor_domestica[eid] = (temporada, liga)

    # Equipos que juegan la competición objetivo en la temporada más reciente.
    fila_temp = con.execute(
        "SELECT MAX(temporada) AS t FROM partidos WHERE competicion = ?", (objetivo,)
    ).fetchone()
    temporada_actual = fila_temp["t"] if fila_temp else None

    en_competicion: set[str] = set()
    if temporada_actual is not None:
        for f in con.execute(
            """SELECT local_id AS a, visitante_id AS b FROM partidos
               WHERE competicion = ? AND temporada = ?""",
            (objetivo, temporada_actual),
        ).fetchall():
            en_competicion.update(x for x in (f["a"], f["b"]) if x)

    registros = [
        {"id": eid,
         "nombre": nombre,
         "liga": mejor_domestica.get(eid, (0, "otras"))[1],
         "en_competicion": 1 if eid in en_competicion else 0}
        for eid, nombre in nombres.items()
    ]

    con.executemany(
        """INSERT INTO equipos (id, nombre, liga, en_competicion)
           VALUES (:id, :nombre, :liga, :en_competicion)
           ON CONFLICT(id) DO UPDATE SET
             nombre         = excluded.nombre,
             liga           = excluded.liga,
             en_competicion = excluded.en_competicion""",
        registros,
    )
    con.commit()
    return len(registros)


def main() -> None:
    ap = argparse.ArgumentParser(description="Descarga el histórico de clubes desde ESPN.")
    ap.add_argument("--desde", help="Fecha mínima 'YYYY-MM-DD' (por defecto, la de config.yaml)")
    ap.add_argument("--solo-actual", action="store_true",
                    help="Solo la temporada en curso (mucho más rápido)")
    ap.add_argument("--competicion", action="append",
                    help="Limitar a un slug concreto (repetible)")
    ap.add_argument("--sin-cache", action="store_true",
                    help="Ignorar la caché en disco y volver a pedirlo todo")
    args = ap.parse_args()

    cfg = cargar_config()
    desde, hasta = _rango_descarga(cfg, args.solo_actual)
    if args.desde:
        desde = datetime.strptime(args.desde, "%Y-%m-%d").date()

    slugs = args.competicion or list(competiciones(cfg).keys())

    con = db.conectar()
    db.inicializar(con)

    print(f"Base:  {ruta_db()}")
    print(f"Rango: {desde} -> {hasta}  ({len(slugs)} competiciones)\n")

    total = 0
    for i, slug in enumerate(slugs, 1):
        etiqueta = competiciones(cfg).get(slug, {}).get("nombre", slug)
        print(f"[{i:2}/{len(slugs)}] {slug:22} {etiqueta:24}", end="", flush=True)
        try:
            partidos = espn.partidos_rango(slug, desde, hasta, usar_cache=not args.sin_cache)
        except espn.ESPNNoDisponible as e:
            print(f"  ⚠️  no disponible en ESPN ({e})")
            continue

        n = guardar_partidos(con, partidos)
        jugados = sum(p["jugado"] for p in partidos)
        total += n
        print(f"  {n:6} partidos ({jugados} jugados)")

    n_equipos = derivar_equipos(con, cfg)
    db.set_meta(con, "ultima_actualizacion", datetime.now().date().isoformat())
    db.set_meta(con, "competicion_objetivo", competicion_id(cfg))

    print(f"\nTotal: {total} partidos, {n_equipos} equipos.")
    en_comp = con.execute(
        "SELECT COUNT(*) n FROM equipos WHERE en_competicion=1").fetchone()["n"]
    print(f"Equipos en {competicion_id(cfg)}: {en_comp}")
    con.close()


if __name__ == "__main__":
    main()
