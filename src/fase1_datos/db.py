"""
db.py — Capa de acceso a la base de datos SQLite del proyecto.

Centraliza el esquema y los helpers de lectura/escritura para que el resto
del código (descarga, modelo, interfaz) no tenga que repetir SQL.

La base activa la decide `config.yaml` (`competicion.bd`):
  - data/clubes.db    -> modo clubes (Champions, LaLiga, Premier...), fuente ESPN
  - data/worldcup.db  -> modo legacy del Mundial 2026, fuente martj42

Tablas principales:
  - partidos       : todos los partidos descargados (histórico + competición
                     objetivo, jugados y futuros). Es la tabla central del modelo.
  - goleadores     : goles individuales (sirve para "primer equipo en anotar").
  - shootouts      : tandas de penales en eliminatorias.
  - equipos        : dimensión de equipos, con la liga a la que pertenecen (la
                     usa el modelo jerárquico) y si están en la competición activa.
  - predicciones   : predicciones generadas por el modelo (para histórico/aciertos).
  - resultados_pred: si cada predicción acertó o no.
"""

from __future__ import annotations  # permite anotaciones tipo 'str | None' en Python 3.9

import sqlite3
from pathlib import Path

from ..config import cargar_config, competicion_id, ruta_db

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]


def conectar(ruta: Path | None = None) -> sqlite3.Connection:
    """Abre (y crea si no existe) la conexión SQLite con buenas opciones por defecto.

    Sin argumento usa la base de la competición activa en config.yaml."""
    ruta = Path(ruta) if ruta else ruta_db()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(ruta))
    con.row_factory = sqlite3.Row          # acceder a columnas por nombre
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")  # mejor concurrencia (actualización diaria)
    return con


# --- Definición del esquema -------------------------------------------------

ESQUEMA = """
CREATE TABLE IF NOT EXISTS partidos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    espn_id         TEXT UNIQUE,                -- id estable del partido en ESPN
    fecha           TEXT    NOT NULL,           -- ISO 'YYYY-MM-DD' (UTC)
    fecha_hora      TEXT,                       -- ISO completo UTC
    local           TEXT    NOT NULL,
    visitante       TEXT    NOT NULL,
    local_id        TEXT,                       -- id ESPN del equipo local
    visitante_id    TEXT,                       -- id ESPN del equipo visitante
    goles_local     INTEGER,                    -- NULL si aún no se juega
    goles_visitante INTEGER,                    -- NULL si aún no se juega
    competicion     TEXT,                       -- slug ESPN: 'uefa.champions', 'esp.1'...
    temporada       INTEGER,                    -- año ESPN: 2026 = 2026-27
    ronda           TEXT,                       -- 'league-phase', 'round-of-16', 'final'...
    leg             INTEGER,                    -- 1 o 2 en eliminatorias a doble partido
    ciudad          TEXT,
    pais            TEXT,                        -- país sede
    neutral         INTEGER DEFAULT 0,          -- 1 si campo neutral
    jugado          INTEGER DEFAULT 0           -- 1 si ya tiene marcador final
);

CREATE INDEX IF NOT EXISTS idx_part_local      ON partidos(local);
CREATE INDEX IF NOT EXISTS idx_part_visitante  ON partidos(visitante);
CREATE INDEX IF NOT EXISTS idx_part_fecha      ON partidos(fecha);
CREATE INDEX IF NOT EXISTS idx_part_comp       ON partidos(competicion, temporada);
CREATE INDEX IF NOT EXISTS idx_part_comp_fecha ON partidos(competicion, fecha);
CREATE INDEX IF NOT EXISTS idx_part_local_id   ON partidos(local_id);
CREATE INDEX IF NOT EXISTS idx_part_visit_id   ON partidos(visitante_id);

CREATE TABLE IF NOT EXISTS goleadores (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    espn_id    TEXT,       -- partido al que pertenece
    fecha      TEXT,
    equipo     TEXT,       -- equipo que anotó
    equipo_id  TEXT,
    jugador    TEXT,
    minuto     INTEGER,
    autogol    INTEGER DEFAULT 0,
    penal      INTEGER DEFAULT 0,
    UNIQUE (espn_id, equipo_id, minuto, jugador)
);
CREATE INDEX IF NOT EXISTS idx_gol_fecha   ON goleadores(fecha);
CREATE INDEX IF NOT EXISTS idx_gol_partido ON goleadores(espn_id);

CREATE TABLE IF NOT EXISTS shootouts (
    espn_id    TEXT PRIMARY KEY,
    fecha      TEXT,
    ganador_id TEXT,
    ganador    TEXT
);

CREATE TABLE IF NOT EXISTS equipos (
    id              TEXT PRIMARY KEY,           -- id ESPN
    nombre          TEXT,
    abreviatura     TEXT,
    liga            TEXT,                       -- clave de agrupación del modelo jerárquico
    en_competicion  INTEGER DEFAULT 0           -- 1 si juega la competición objetivo
);
CREATE INDEX IF NOT EXISTS idx_eq_nombre ON equipos(nombre);
CREATE INDEX IF NOT EXISTS idx_eq_liga   ON equipos(liga);

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


def calendario_de_fecha(con: sqlite3.Connection, fecha: str, comp: str | None = None):
    """Partidos de la competición objetivo programados para una fecha concreta."""
    comp = comp or competicion_id()
    return con.execute(
        "SELECT * FROM partidos WHERE competicion = ? AND fecha = ? "
        "ORDER BY fecha_hora, id",
        (comp, fecha),
    ).fetchall()


def fechas_competicion(con: sqlite3.Connection, comp: str | None = None,
                       temporada: int | None = None) -> list[str]:
    """Todas las fechas con partidos de la competición objetivo, ordenadas."""
    comp = comp or competicion_id()
    sql = "SELECT DISTINCT fecha FROM partidos WHERE competicion = ?"
    params: list = [comp]
    if temporada is not None:
        sql += " AND temporada = ?"
        params.append(temporada)
    sql += " ORDER BY fecha"
    return [f["fecha"] for f in con.execute(sql, params).fetchall()]


def num_partidos_equipo(con: sqlite3.Connection, equipo: str, desde: str | None = None) -> int:
    """Cuántos partidos jugados tiene un equipo (para calcular nivel de confianza)."""
    sql = ("SELECT COUNT(*) AS n FROM partidos "
           "WHERE jugado = 1 AND (local = ? OR visitante = ?)")
    params = [equipo, equipo]
    if desde:
        sql += " AND fecha >= ?"
        params.append(desde)
    return con.execute(sql, params).fetchone()["n"]


def equipos_competicion(con: sqlite3.Connection) -> list[str]:
    """Equipos que disputan la competición objetivo (derivado de los datos)."""
    filas = con.execute(
        "SELECT nombre FROM equipos WHERE en_competicion = 1 ORDER BY nombre"
    ).fetchall()
    return [f["nombre"] for f in filas]


def liga_por_equipo(con: sqlite3.Connection) -> dict[str, str]:
    """nombre de equipo -> clave de liga. Es la entrada del modelo jerárquico."""
    filas = con.execute("SELECT nombre, liga FROM equipos").fetchall()
    return {f["nombre"]: (f["liga"] or "otras") for f in filas}


if __name__ == "__main__":
    # Permite crear la BD vacía con: python -m src.fase1_datos.db
    con = conectar()
    inicializar(con)
    print(f"Base de datos inicializada en: {ruta_db()}")
    con.close()
