"""
db.py — Capa de acceso a la base de datos SQLite del proyecto.

Centraliza el esquema y los helpers de lectura/escritura para que el resto
del código (descarga, modelo, interfaz) no tenga que repetir SQL.

Tablas principales:
  - partidos      : todos los partidos internacionales (histórico + Mundial 2026,
                    jugados y futuros). Es la tabla central del modelo.
  - goleadores    : goles individuales (sirve para "primer equipo en anotar").
  - shootouts     : tandas de penales en eliminatorias.
  - equipos       : dimensión de equipos con metadatos (confederación, si está
                    clasificado al Mundial 2026).
  - predicciones  : predicciones generadas por el modelo (para histórico/aciertos).
  - resultados_pred: si cada predicción acertó o no (alimenta el backtesting/tracking).
"""

from __future__ import annotations  # permite anotaciones tipo 'str | None' en Python 3.9

import sqlite3
from pathlib import Path

# Ruta a la base de datos (data/worldcup.db en la raíz del proyecto)
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_DB = RAIZ_PROYECTO / "data" / "worldcup.db"


def conectar(ruta_db: Path = RUTA_DB) -> sqlite3.Connection:
    """Abre (y crea si no existe) la conexión SQLite con buenas opciones por defecto."""
    ruta_db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(ruta_db))
    con.row_factory = sqlite3.Row          # acceder a columnas por nombre
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")  # mejor concurrencia (actualización diaria)
    return con


# --- Definición del esquema -------------------------------------------------

ESQUEMA = """
CREATE TABLE IF NOT EXISTS partidos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha           TEXT    NOT NULL,           -- ISO 'YYYY-MM-DD'
    local           TEXT    NOT NULL,
    visitante       TEXT    NOT NULL,
    goles_local     INTEGER,                    -- NULL si aún no se juega
    goles_visitante INTEGER,                    -- NULL si aún no se juega
    torneo          TEXT,
    ciudad          TEXT,
    pais            TEXT,                        -- país sede
    neutral         INTEGER DEFAULT 0,          -- 1 si campo neutral
    jugado          INTEGER DEFAULT 0,          -- 1 si ya tiene marcador
    es_mundial2026  INTEGER DEFAULT 0,          -- 1 si es partido del Mundial 2026
    UNIQUE (fecha, local, visitante)
);

CREATE INDEX IF NOT EXISTS idx_part_local     ON partidos(local);
CREATE INDEX IF NOT EXISTS idx_part_visitante ON partidos(visitante);
CREATE INDEX IF NOT EXISTS idx_part_fecha     ON partidos(fecha);
CREATE INDEX IF NOT EXISTS idx_part_wc2026    ON partidos(es_mundial2026);

CREATE TABLE IF NOT EXISTS goleadores (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha     TEXT,
    local     TEXT,
    visitante TEXT,
    equipo    TEXT,      -- equipo que anotó
    jugador   TEXT,
    minuto    INTEGER,
    autogol   INTEGER DEFAULT 0,
    penal     INTEGER DEFAULT 0,
    UNIQUE (fecha, local, visitante, jugador, minuto, equipo)
);
CREATE INDEX IF NOT EXISTS idx_gol_fecha ON goleadores(fecha);

CREATE TABLE IF NOT EXISTS shootouts (
    fecha     TEXT,
    local     TEXT,
    visitante TEXT,
    ganador   TEXT,
    UNIQUE (fecha, local, visitante)
);

CREATE TABLE IF NOT EXISTS equipos (
    nombre           TEXT PRIMARY KEY,
    confederacion    TEXT,
    clasificado_2026 INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS predicciones (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    partido_id    INTEGER REFERENCES partidos(id),
    fecha_pred    TEXT,              -- cuándo se generó la predicción
    mercado       TEXT,              -- '1X2', 'OU2.5', 'BTTS', 'marcador_exacto', ...
    seleccion     TEXT,              -- 'Local', 'Over', 'Si', '2-1', ...
    probabilidad  REAL,              -- 0..1
    confianza     TEXT               -- 'alto' | 'medio' | 'bajo'
);
CREATE INDEX IF NOT EXISTS idx_pred_partido ON predicciones(partido_id);

CREATE TABLE IF NOT EXISTS resultados_pred (
    prediccion_id INTEGER PRIMARY KEY REFERENCES predicciones(id),
    acerto        INTEGER            -- 1 acertó, 0 falló, NULL pendiente
);

-- Metadatos de la base (ej. fecha de última actualización)
CREATE TABLE IF NOT EXISTS meta (
    clave TEXT PRIMARY KEY,
    valor TEXT
);
"""


def inicializar(con: sqlite3.Connection) -> None:
    """Crea todas las tablas e índices si no existen."""
    con.executescript(ESQUEMA)
    con.commit()


def set_meta(con: sqlite3.Connection, clave: str, valor: str) -> None:
    con.execute(
        "INSERT INTO meta(clave, valor) VALUES(?, ?) "
        "ON CONFLICT(clave) DO UPDATE SET valor=excluded.valor",
        (clave, valor),
    )
    con.commit()


def get_meta(con: sqlite3.Connection, clave: str, defecto=None):
    fila = con.execute("SELECT valor FROM meta WHERE clave=?", (clave,)).fetchone()
    return fila["valor"] if fila else defecto


# --- Helpers de lectura usados por el modelo --------------------------------

def partidos_jugados(con: sqlite3.Connection, desde: str | None = None):
    """Devuelve todos los partidos ya jugados (con marcador), opcionalmente
    filtrando por fecha mínima 'YYYY-MM-DD'. Lista de sqlite3.Row."""
    sql = "SELECT * FROM partidos WHERE jugado = 1"
    params: list = []
    if desde:
        sql += " AND fecha >= ?"
        params.append(desde)
    sql += " ORDER BY fecha"
    return con.execute(sql, params).fetchall()


def calendario_de_fecha(con: sqlite3.Connection, fecha: str):
    """Partidos del Mundial 2026 programados para una fecha concreta."""
    return con.execute(
        "SELECT * FROM partidos WHERE es_mundial2026 = 1 AND fecha = ? ORDER BY id",
        (fecha,),
    ).fetchall()


def num_partidos_equipo(con: sqlite3.Connection, equipo: str, desde: str | None = None) -> int:
    """Cuántos partidos jugados tiene un equipo (para calcular nivel de confianza)."""
    sql = ("SELECT COUNT(*) AS n FROM partidos "
           "WHERE jugado = 1 AND (local = ? OR visitante = ?)")
    params = [equipo, equipo]
    if desde:
        sql += " AND fecha >= ?"
        params.append(desde)
    return con.execute(sql, params).fetchone()["n"]


def equipos_mundial2026(con: sqlite3.Connection) -> list[str]:
    """Lista de selecciones que disputan el Mundial 2026 (derivada de los datos)."""
    filas = con.execute(
        "SELECT nombre FROM equipos WHERE clasificado_2026 = 1 ORDER BY nombre"
    ).fetchall()
    return [f["nombre"] for f in filas]


if __name__ == "__main__":
    # Permite crear la BD vacía con: python -m src.fase1_datos.db
    con = conectar()
    inicializar(con)
    print(f"Base de datos inicializada en: {RUTA_DB}")
    con.close()
