"""
backtest.py — Validación honesta del modelo contra torneos pasados.

Entrena el modelo SOLO con datos anteriores al torneo objetivo (sin filtrar el
futuro) y predice cada partido del torneo, comparando con el resultado real.

Métricas (cuanto MÁS BAJO mejor, salvo accuracy):
  - accuracy 1X2  : % de aciertos del resultado más probable (argmax).
  - log-loss      : penaliza dar baja probabilidad al resultado real (calibración).
  - RPS           : Ranked Probability Score, el estándar en fútbol para 1X2
                    ordenado (local/empate/visitante); premia repartir bien la
                    probabilidad incluso cuando falla el argmax.
  - Brier (BTTS / Over2.5): error cuadrático de los mercados binarios.

Compara contra una LÍNEA BASE (las frecuencias históricas locales/empate/visita),
para demostrar que el modelo aporta valor real y no solo "adivina la media".

Uso:
    python -m src.backtesting.backtest                 # backtest por defecto (varios torneos)
    python -m src.backtesting.backtest --tune          # + búsqueda de hiperparámetros
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from ..fase1_datos import db
from ..fase2_modelo import mercados
from ..fase2_modelo.entrenar import cargar_config, entrenar_modelo


# Torneos a validar: (etiqueta, nombre_en_datos, desde, hasta)
TORNEOS_DEFECTO = [
    ("Mundial 2018", "FIFA World Cup", "2018-06-01", "2018-07-31"),
    ("Mundial 2022", "FIFA World Cup", "2022-11-01", "2022-12-31"),
    ("Eurocopa 2021", "UEFA Euro", "2021-06-01", "2021-07-31"),
    ("Eurocopa 2024", "UEFA Euro", "2024-06-01", "2024-07-31"),
]


def _rps_1x2(probs: list, resultado: int) -> float:
    """RPS para 3 resultados ordenados [local, empate, visitante].
    'resultado' es 0/1/2. Devuelve el score (menor = mejor)."""
    obs = [0.0, 0.0, 0.0]
    obs[resultado] = 1.0
    acum_p = acum_o = 0.0
    s = 0.0
    for i in range(2):  # r-1 = 2 términos
        acum_p += probs[i]
        acum_o += obs[i]
        s += (acum_p - acum_o) ** 2
    return s / 2.0


def backtest_evento(con, cfg, nombre, torneo, desde, hasta, verbose=True):
    """Backtest de un torneo. Devuelve dict de métricas."""
    # Partidos jugados del torneo en la ventana de fechas
    partidos = con.execute(
        """SELECT local, visitante, goles_local, goles_visitante, neutral
             FROM partidos
            WHERE jugado=1 AND torneo=? AND fecha>=? AND fecha<=?
            ORDER BY fecha""",
        (torneo, desde, hasta),
    ).fetchall()
    if not partidos:
        return None

    # Entrenar SOLO con datos anteriores al torneo (corte = 'desde')
    modelo = entrenar_modelo(con, cfg, hasta=desde)

    # Línea base: frecuencias de resultado en los datos de entrenamiento
    base = con.execute(
        """SELECT
             AVG(goles_local > goles_visitante) AS l,
             AVG(goles_local = goles_visitante) AS e,
             AVG(goles_local < goles_visitante) AS v
           FROM partidos WHERE jugado=1 AND fecha < ?""",
        (desde,),
    ).fetchone()
    base_probs = [base["l"], base["e"], base["v"]]

    n = 0
    aciertos = 0
    logloss = rps = brier_btts = brier_ou = 0.0
    base_logloss = base_rps = 0.0
    saltados = 0

    for p in partidos:
        loc, vis = p["local"], p["visitante"]
        if loc not in modelo.idx or vis not in modelo.idx:
            saltados += 1
            continue
        M = modelo.matriz_marcador(loc, vis, p["neutral"])
        x1 = mercados.resultado_1x2(M)
        probs = [x1["local"], x1["empate"], x1["visitante"]]

        gl, gv = p["goles_local"], p["goles_visitante"]
        real = 0 if gl > gv else (1 if gl == gv else 2)

        # Métricas 1X2
        if int(np.argmax(probs)) == real:
            aciertos += 1
        logloss += -math.log(max(probs[real], 1e-12))
        rps += _rps_1x2(probs, real)
        base_logloss += -math.log(max(base_probs[real], 1e-12))
        base_rps += _rps_1x2(base_probs, real)

        # Mercados binarios
        btts = mercados.btts(M)["si"]
        btts_real = 1.0 if (gl >= 1 and gv >= 1) else 0.0
        brier_btts += (btts - btts_real) ** 2
        over = mercados.over_under(M, (2.5,))["over_2.5"]
        over_real = 1.0 if (gl + gv) > 2.5 else 0.0
        brier_ou += (over - over_real) ** 2

        n += 1

    if n == 0:
        return None
    res = {
        "nombre": nombre, "n": n, "saltados": saltados,
        "accuracy": aciertos / n,
        "logloss": logloss / n,
        "rps": rps / n,
        "brier_btts": brier_btts / n,
        "brier_ou": brier_ou / n,
        "base_logloss": base_logloss / n,
        "base_rps": base_rps / n,
    }
    if verbose:
        print(f"\n  {nombre}  ({n} partidos, {saltados} sin datos previos)")
        print(f"    Accuracy 1X2 : {res['accuracy']*100:5.1f}%")
        print(f"    Log-loss     : {res['logloss']:.4f}   (base {res['base_logloss']:.4f})")
        print(f"    RPS          : {res['rps']:.4f}   (base {res['base_rps']:.4f})")
        print(f"    Brier BTTS   : {res['brier_btts']:.4f}")
        print(f"    Brier O/U2.5 : {res['brier_ou']:.4f}")
    return res


def correr_backtest(con, cfg, torneos=None, verbose=True):
    torneos = torneos or TORNEOS_DEFECTO
    resultados = []
    for nombre, torneo, desde, hasta in torneos:
        r = backtest_evento(con, cfg, nombre, torneo, desde, hasta, verbose)
        if r:
            resultados.append(r)
    if resultados and verbose:
        n_tot = sum(r["n"] for r in resultados)
        acc = sum(r["accuracy"] * r["n"] for r in resultados) / n_tot
        ll = sum(r["logloss"] * r["n"] for r in resultados) / n_tot
        rps = sum(r["rps"] * r["n"] for r in resultados) / n_tot
        bll = sum(r["base_logloss"] * r["n"] for r in resultados) / n_tot
        brps = sum(r["base_rps"] * r["n"] for r in resultados) / n_tot
        print(f"\n  === GLOBAL ({n_tot} partidos) ===")
        print(f"    Accuracy : {acc*100:5.1f}%")
        print(f"    Log-loss : {ll:.4f}   (base {bll:.4f}, mejora {(bll-ll)/bll*100:4.1f}%)")
        print(f"    RPS      : {rps:.4f}   (base {brps:.4f}, mejora {(brps-rps)/brps*100:4.1f}%)")
    return resultados


def tune_hiperparametros(con, cfg):
    """Busca (vida_media, reg) que minimicen el log-loss medio en backtest.
    Esto es lo que vuelve el modelo lo MÁS PRECISO posible con estos datos."""
    print("\nBúsqueda de hiperparámetros (esto entrena el modelo varias veces)...")
    mejor = None
    for vm in [1.0, 1.5, 2.0, 3.0]:
        for reg in [0.5, 1.0, 2.0, 4.0]:
            cfg["modelo"]["vida_media_anios"] = vm
            cfg["modelo"]["reg"] = reg
            resultados = correr_backtest(con, cfg, verbose=False)
            n_tot = sum(r["n"] for r in resultados)
            ll = sum(r["logloss"] * r["n"] for r in resultados) / n_tot
            rps = sum(r["rps"] * r["n"] for r in resultados) / n_tot
            print(f"    vida_media={vm}  reg={reg}  ->  logloss={ll:.4f}  rps={rps:.4f}")
            if mejor is None or ll < mejor[0]:
                mejor = (ll, rps, vm, reg)
    print(f"\n  MEJOR: vida_media={mejor[2]}  reg={mejor[3]}  "
          f"(logloss={mejor[0]:.4f}, rps={mejor[1]:.4f})")
    print("  -> Considera fijar estos valores en config.yaml")
    return mejor


def main():
    parser = argparse.ArgumentParser(description="Backtesting del modelo.")
    parser.add_argument("--tune", action="store_true",
                        help="Buscar mejores hiperparámetros (más lento).")
    args = parser.parse_args()

    con = db.conectar()
    cfg = cargar_config()
    print("BACKTESTING — validando el modelo contra torneos pasados")
    print("(se entrena solo con datos anteriores a cada torneo: sin trampa)")
    correr_backtest(con, cfg)
    if args.tune:
        tune_hiperparametros(con, cfg)
    con.close()


if __name__ == "__main__":
    main()
