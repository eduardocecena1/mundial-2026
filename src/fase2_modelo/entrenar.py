"""
entrenar.py — Carga datos desde la BD, calcula pesos y entrena el Dixon-Coles.

Separa la lógica de "preparar datos + ajustar el modelo" para que la usen tanto
la predicción diaria (Fase 4) como el backtesting (entrenando con un corte de
fecha que excluye la ventana que se quiere validar).

Incluye una caché de modelos en disco: `seguimiento.evaluar_rango()` reentrena
una vez por cada fecha jugada y, con decenas de miles de partidos de clubes,
cada ajuste ya no es instantáneo. La clave de caché incluye los hiperparámetros,
así que tocar `config.yaml` invalida lo cacheado automáticamente.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from ..config import cargar_config, competiciones, liga_de_competicion  # noqa: F401 (re-export)
from ..fase1_datos import db
from .poisson_dixon_coles import DixonColes, peso_temporal, peso_torneo

DIR_CACHE_MODELOS = db.RAIZ_PROYECTO / "data" / "processed" / "modelos"


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
    honesto (no se entrena con el futuro).

    Devuelve además la liga de cada partido y el mapa equipo -> liga, que son
    las dos entradas nuevas del modelo jerárquico.
    """
    desde = cfg["datos"]["desde"]
    vida_media = cfg["modelo"]["vida_media_anios"]
    pesos_cfg = cfg["datos"].get("competiciones", {})

    filas = con.execute(
        """SELECT fecha, local, visitante, goles_local, goles_visitante,
                  competicion, neutral
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
    comps = [f["competicion"] for f in filas]

    # Peso = decaimiento temporal * importancia de la competición
    edad = _anios_entre(fechas, hasta)
    w_tiempo = peso_temporal(edad, vida_media)
    w_comp = np.array([peso_torneo(t, pesos_cfg) for t in comps])
    peso = w_tiempo * w_comp

    # Modelo jerárquico: liga de cada equipo y liga de cada partido
    liga_equipo = db.liga_por_equipo(con)
    liga_partido = np.array([liga_de_competicion(c, cfg) for c in comps])

    return local, visit, gl, gv, neutral, peso, liga_equipo, liga_partido


def _clave_cache(cfg: dict, hasta: str) -> str:
    """Hash de los hiperparámetros + corte de fecha + competición activa."""
    relevante = {
        "hasta": hasta,
        "desde": cfg["datos"]["desde"],
        "modelo": cfg["modelo"],
        "competiciones": cfg["datos"].get("competiciones"),
        "bd": str(db.ruta_db()),
    }
    bruto = json.dumps(relevante, sort_keys=True, default=str)
    return hashlib.sha1(bruto.encode()).hexdigest()[:16]


def entrenar_modelo(con, cfg: dict, hasta: str | None = None,
                    usar_cache: bool = True) -> DixonColes:
    """Entrena y devuelve un DixonColes ajustado con los datos hasta 'hasta'
    (por defecto, mañana: usa todo lo jugado hasta hoy)."""
    if hasta is None:
        # incluir todo lo jugado hasta hoy -> corte = mañana
        hasta = (date.today() + timedelta(days=1)).isoformat()

    ruta_cache = DIR_CACHE_MODELOS / f"{_clave_cache(cfg, hasta)}.pkl"
    if usar_cache and ruta_cache.exists():
        try:
            with open(ruta_cache, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError):
            ruta_cache.unlink()          # caché corrupta o de otra versión del código

    local, visit, gl, gv, neutral, peso, liga_equipo, liga_partido = preparar_datos(
        con, cfg, hasta)

    modelo = DixonColes(
        max_goles=cfg["modelo"]["max_goles"],
        reg=cfg["modelo"].get("reg", 1.0),
        reg_liga=cfg["modelo"].get("reg_liga", 0.05),
    )
    modelo.fit(local, visit, gl, gv, neutral, peso,
               liga_equipo=liga_equipo, liga_partido=liga_partido)

    if usar_cache:
        ruta_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta_cache, "wb") as f:
            pickle.dump(modelo, f)
    return modelo


def limpiar_cache_modelos() -> int:
    """Borra los modelos cacheados. Útil tras cambiar el código del modelo."""
    if not DIR_CACHE_MODELOS.exists():
        return 0
    n = 0
    for p in DIR_CACHE_MODELOS.glob("*.pkl"):
        p.unlink()
        n += 1
    return n


if __name__ == "__main__":
    import time

    from ..config import nombre_competicion

    con = db.conectar()
    cfg = cargar_config()
    print(f"Entrenando Dixon-Coles jerárquico — {nombre_competicion(cfg)}")
    t0 = time.time()
    modelo = entrenar_modelo(con, cfg, usar_cache=False)
    print(f"Ajustado en {time.time() - t0:.1f}s  "
          f"({len(modelo.equipos)} equipos, {len(modelo.ligas)} ligas)")

    print(f"\nGlobales: intercepto={modelo.intercepto:+.3f}  rho={modelo.rho:+.3f}")

    print("\nEfecto de LIGA (ataque = cuánto marca de más esa liga; "
          "defensa = cuánto encaja de menos):")
    fuerzas = modelo.fuerza_ligas()
    for liga, v in sorted(fuerzas.items(), key=lambda kv: -(kv[1]["ataque"] + kv[1]["defensa"])):
        n_eq = sum(1 for l in modelo.liga_equipo.values() if l == liga)
        print(f"  {liga:8s} atk={v['ataque']:+.3f}  def={v['defensa']:+.3f}  "
              f"local={v['ventaja_local']:+.3f}  ({n_eq} equipos)")

    # Ojo al mostrar el ranking: un recién ascendido con 3 partidos jugados puede
    # colarse arriba. El ridge lo encoge en proporción a sus datos, pero 3 partidos
    # dan para poco. Por eso el ranking va SIEMPRE con el nº de partidos detrás, y
    # además se lista aparte la competición objetivo, que es lo que se predice.
    n_partidos = {e: db.num_partidos_equipo(con, e, cfg["datos"]["desde"])
                  for e in modelo.equipos}
    neta = modelo.ataque + modelo.defensa
    MIN_FIABLE = cfg.get("confianza", {}).get("medio", 40)

    print(f"\nTop 15 por fuerza NETA (solo equipos con >= {MIN_FIABLE} partidos):")
    fiables = [i for i in np.argsort(neta)[::-1]
               if n_partidos[modelo.equipos[i]] >= MIN_FIABLE]
    for i in fiables[:15]:
        e = modelo.equipos[i]
        print(f"  {e:26s} [{modelo.liga_equipo.get(e, '?'):5s}] "
              f"atk={modelo.ataque[i]:+.2f}  def={modelo.defensa[i]:+.2f}  "
              f"neta={neta[i]:+.2f}  ({n_partidos[e]} part.)")

    escasos = [i for i in np.argsort(neta)[::-1][:40]
               if n_partidos[modelo.equipos[i]] < MIN_FIABLE]
    if escasos:
        print("\n  Con pocos datos y estimación alta (tómalas con pinzas):")
        for i in escasos[:5]:
            e = modelo.equipos[i]
            print(f"    {e:26s} neta={neta[i]:+.2f}  ({n_partidos[e]} part.)")

    print(f"\nLos {len(db.equipos_competicion(con))} equipos de "
          f"{nombre_competicion(cfg)}, por fuerza neta:")
    en_comp = [(e, neta[modelo.idx[e]]) for e in db.equipos_competicion(con)
               if e in modelo.idx]
    for pos, (e, v) in enumerate(sorted(en_comp, key=lambda kv: -kv[1]), 1):
        marca = "  ⚠️ pocos datos" if n_partidos[e] < MIN_FIABLE else ""
        print(f"  {pos:2}. {e:26s} [{modelo.liga_equipo.get(e, '?'):5s}] "
              f"neta={v:+.2f}  ({n_partidos[e]} part.){marca}")
    con.close()
