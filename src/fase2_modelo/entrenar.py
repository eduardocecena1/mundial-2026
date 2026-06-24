"""
entrenar.py — Carga datos desde la BD, calcula pesos y entrena el Dixon-Coles.

Separa la lógica de "preparar datos + ajustar el modelo" para que la usen tanto
la predicción diaria (Fase 4) como el backtesting (entrenando con un corte de
fecha que excluye el torneo que se quiere validar).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import yaml

from ..fase1_datos import db
from .poisson_dixon_coles import DixonColes, peso_torneo, peso_temporal


def cargar_config(ruta=None) -> dict:
    """Lee config.yaml (parámetros del modelo y de las apuestas)."""
    ruta = ruta or (db.RAIZ_PROYECTO / "config.yaml")
    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _anios_entre(fechas_iso: np.ndarray, ref: str) -> np.ndarray:
    """Edad en años (float) de cada partido respecto a la fecha de referencia."""
    ref_d = date.fromisoformat(ref)
    out = np.empty(len(fechas_iso))
    for i, f in enumerate(fechas_iso):
        d = date.fromisoformat(f)
        out[i] = (ref_d - d).days / 365.25
    return out


def preparar_datos(con, cfg: dict, hasta: str):
    """Lee los partidos jugados ANTES de 'hasta' y devuelve los arrays + pesos
    listos para entrenar. Excluir partidos desde 'hasta' permite backtesting
    honesto (no se entrena con el futuro)."""
    desde = cfg["datos"]["desde"]
    vida_media = cfg["modelo"]["vida_media_anios"]
    pesos_torneo_cfg = cfg["modelo"]["pesos_torneo"]

    filas = con.execute(
        """SELECT fecha, local, visitante, goles_local, goles_visitante,
                  torneo, neutral
             FROM partidos
            WHERE jugado = 1 AND fecha >= ? AND fecha < ?
            ORDER BY fecha""",
        (desde, hasta),
    ).fetchall()

    local = np.array([f["local"] for f in filas])
    visit = np.array([f["visitante"] for f in filas])
    gl = np.array([f["goles_local"] for f in filas])
    gv = np.array([f["goles_visitante"] for f in filas])
    neutral = np.array([f["neutral"] for f in filas])
    fechas = np.array([f["fecha"] for f in filas])
    torneos = [f["torneo"] for f in filas]

    # Peso = decaimiento temporal * importancia del torneo
    edad = _anios_entre(fechas, hasta)
    w_tiempo = peso_temporal(edad, vida_media)
    w_torneo = np.array([peso_torneo(t, pesos_torneo_cfg) for t in torneos])
    peso = w_tiempo * w_torneo

    return local, visit, gl, gv, neutral, peso


def entrenar_modelo(con, cfg: dict, hasta: str | None = None) -> DixonColes:
    """Entrena y devuelve un DixonColes ajustado con los datos hasta 'hasta'
    (por defecto, mañana: usa todo lo jugado hasta hoy)."""
    if hasta is None:
        # incluir todo lo jugado hasta hoy -> corte = mañana
        from datetime import timedelta
        hasta = (date.today() + timedelta(days=1)).isoformat()

    local, visit, gl, gv, neutral, peso = preparar_datos(con, cfg, hasta)
    modelo = DixonColes(
        max_goles=cfg["modelo"]["max_goles"],
        reg=cfg["modelo"].get("reg", 1.0),
    )
    modelo.fit(local, visit, gl, gv, neutral, peso)
    return modelo


if __name__ == "__main__":
    # Prueba rápida: entrenar y mostrar las 10 selecciones más fuertes en ataque.
    con = db.conectar()
    cfg = cargar_config()
    print("Entrenando Dixon-Coles con todo el histórico disponible...")
    modelo = entrenar_modelo(con, cfg)
    orden = np.argsort(modelo.ataque)[::-1]
    print(f"\nParámetros globales: intercepto={modelo.intercepto:.3f}  "
          f"ventaja_local={modelo.ventaja_local:.3f}  rho={modelo.rho:.3f}")
    print("\nTop 12 selecciones por fuerza de ATAQUE (ataque / defensa):")
    for i in orden[:12]:
        print(f"  {modelo.equipos[i]:24s} atk={modelo.ataque[i]:+.2f}  "
              f"def={modelo.defensa[i]:+.2f}")
    con.close()
