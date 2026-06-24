"""
descargar_historico.py — FASE 1: recolección de datos.

Descarga el dataset público y vivo de resultados internacionales mantenido por
martj42 (la misma base que el dataset de Kaggle "International football results",
pero actualizada a diario en GitHub e incluyendo YA los partidos del Mundial 2026,
jugados y futuros). Lo carga en la base SQLite del proyecto.

Fuentes (CSV, sin necesidad de API key):
  - results.csv     : todos los partidos internacionales (1872 -> presente)
  - goalscorers.csv : goleadores por partido (minuto, penal, autogol)
  - shootouts.csv   : tandas de penales

Uso:
    python -m src.fase1_datos.descargar_historico
    python -m src.fase1_datos.descargar_historico --desde 2014-01-01
"""

import argparse
import io
from datetime import date

import pandas as pd
import requests

from . import db

BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
ARCHIVOS = {
    "results": f"{BASE_URL}/results.csv",
    "goalscorers": f"{BASE_URL}/goalscorers.csv",
    "shootouts": f"{BASE_URL}/shootouts.csv",
}

# Por defecto cargamos desde 2014 (cubre 3 ciclos mundialistas, como pidió el proyecto).
# El histórico completo se guarda en raw, pero la BD del modelo arranca aquí.
DESDE_DEFECTO = "2014-01-01"


def _descargar_csv(url: str) -> pd.DataFrame:
    """Descarga un CSV remoto a un DataFrame, guardando copia en data/raw/."""
    print(f"  Descargando {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    nombre = url.rsplit("/", 1)[-1]
    (db.RAIZ_PROYECTO / "data" / "raw" / nombre).write_bytes(resp.content)
    return pd.read_csv(io.StringIO(resp.text))


def _a_int(valor):
    """Convierte marcador a int o None (los partidos futuros vienen como 'NA')."""
    if pd.isna(valor):
        return None
    try:
        return int(float(valor))
    except (ValueError, TypeError):
        return None


def cargar_results(con, df: pd.DataFrame, desde: str) -> int:
    """Inserta/actualiza partidos. Marca jugados y partidos del Mundial 2026."""
    df = df[df["date"] >= desde].copy()
    filas = []
    for r in df.itertuples(index=False):
        gl, gv = _a_int(r.home_score), _a_int(r.away_score)
        jugado = 1 if (gl is not None and gv is not None) else 0
        es_wc2026 = 1 if (str(r.tournament) == "FIFA World Cup"
                          and str(r.date) >= "2026-01-01") else 0
        neutral = 1 if str(r.neutral).strip().upper() == "TRUE" else 0
        filas.append((
            str(r.date), str(r.home_team), str(r.away_team), gl, gv,
            str(r.tournament), str(r.city), str(r.country),
            neutral, jugado, es_wc2026,
        ))

    # UPSERT: si el partido ya existe (misma fecha/local/visitante) actualiza el
    # marcador. Así la actualización diaria rellena resultados de partidos futuros.
    con.executemany(
        """INSERT INTO partidos
             (fecha, local, visitante, goles_local, goles_visitante,
              torneo, ciudad, pais, neutral, jugado, es_mundial2026)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(fecha, local, visitante) DO UPDATE SET
             goles_local     = excluded.goles_local,
             goles_visitante = excluded.goles_visitante,
             jugado          = excluded.jugado,
             es_mundial2026  = excluded.es_mundial2026""",
        filas,
    )
    con.commit()
    return len(filas)


def cargar_goleadores(con, df: pd.DataFrame, desde: str) -> int:
    df = df[df["date"] >= desde].copy()
    filas = []
    for r in df.itertuples(index=False):
        filas.append((
            str(r.date), str(r.home_team), str(r.away_team), str(r.team),
            str(r.scorer) if not pd.isna(r.scorer) else None,
            _a_int(r.minute),
            1 if str(r.own_goal).strip().upper() == "TRUE" else 0,
            1 if str(r.penalty).strip().upper() == "TRUE" else 0,
        ))
    con.executemany(
        """INSERT OR IGNORE INTO goleadores
             (fecha, local, visitante, equipo, jugador, minuto, autogol, penal)
           VALUES (?,?,?,?,?,?,?,?)""",
        filas,
    )
    con.commit()
    return len(filas)


def cargar_shootouts(con, df: pd.DataFrame, desde: str) -> int:
    df = df[df["date"] >= desde].copy()
    filas = [(str(r.date), str(r.home_team), str(r.away_team), str(r.winner))
             for r in df.itertuples(index=False)]
    con.executemany(
        """INSERT OR IGNORE INTO shootouts (fecha, local, visitante, ganador)
           VALUES (?,?,?,?)""",
        filas,
    )
    con.commit()
    return len(filas)


def derivar_equipos(con) -> None:
    """Construye la tabla 'equipos' a partir de los datos:
       - todos los equipos que aparecen en partidos
       - marca clasificado_2026 = 1 si juega algún partido del Mundial 2026.
    Así la lista de participantes sale de datos reales, no de memoria."""
    con.execute("DELETE FROM equipos")
    con.execute(
        """INSERT INTO equipos (nombre, clasificado_2026)
           SELECT nombre, MAX(es_wc) FROM (
               SELECT local AS nombre, es_mundial2026 AS es_wc FROM partidos
               UNION ALL
               SELECT visitante AS nombre, es_mundial2026 AS es_wc FROM partidos
           ) GROUP BY nombre"""
    )
    con.commit()


def main():
    parser = argparse.ArgumentParser(description="Descarga el histórico a SQLite.")
    parser.add_argument("--desde", default=DESDE_DEFECTO,
                        help="Fecha mínima YYYY-MM-DD (defecto 2014-01-01).")
    args = parser.parse_args()

    con = db.conectar()
    db.inicializar(con)

    print("FASE 1 — Descargando datos históricos e internacionales...")
    df_results = _descargar_csv(ARCHIVOS["results"])
    df_goles = _descargar_csv(ARCHIVOS["goalscorers"])
    df_shoot = _descargar_csv(ARCHIVOS["shootouts"])

    print(f"\nCargando en la base de datos (desde {args.desde})...")
    n_p = cargar_results(con, df_results, args.desde)
    n_g = cargar_goleadores(con, df_goles, args.desde)
    n_s = cargar_shootouts(con, df_shoot, args.desde)
    derivar_equipos(con)
    db.set_meta(con, "ultima_actualizacion", date.today().isoformat())

    # Resumen
    jugados = con.execute("SELECT COUNT(*) AS n FROM partidos WHERE jugado=1").fetchone()["n"]
    wc = con.execute("SELECT COUNT(*) AS n FROM partidos WHERE es_mundial2026=1").fetchone()["n"]
    clasif = con.execute("SELECT COUNT(*) AS n FROM equipos WHERE clasificado_2026=1").fetchone()["n"]

    print("\n=== RESUMEN DE CARGA ===")
    print(f"  Partidos cargados (>= {args.desde}): {n_p}  (jugados: {jugados})")
    print(f"  Goles individuales cargados        : {n_g}")
    print(f"  Tandas de penales cargadas         : {n_s}")
    print(f"  Partidos del Mundial 2026          : {wc}")
    print(f"  Selecciones en el Mundial 2026     : {clasif}")
    print(f"\nBase de datos lista en: {db.RUTA_DB}")
    con.close()


if __name__ == "__main__":
    main()
