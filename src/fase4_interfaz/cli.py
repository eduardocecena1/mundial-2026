"""
cli.py — FASE 4: interfaz de línea de comandos.

Imprime un reporte limpio y ordenado de las predicciones y las 3 Leyes para una
fecha de la competición activa (por defecto, la Champions).

Uso (desde la raíz del proyecto):
    python predicciones.py --fecha 2026-09-30
    python predicciones.py --fecha 2026-09-30 --actualizar   # baja datos frescos antes
    python predicciones.py --fecha 2026-09-30 --detalle      # + tabla por partido
    python predicciones.py --fecha 2027-02-17 --eliminatorias # probabilidades de pase
"""

from __future__ import annotations

import argparse
from datetime import date

from ..config import (cargar_config, competicion_id, liga_de_competicion,
                      nombre_competicion)
from ..fase1_datos import db
from ..fase2_modelo import eliminatoria as elim
from ..fase2_modelo.entrenar import entrenar_modelo
from ..fase2_modelo.predecir_partido import imprimir_prediccion, predecir
from ..fase3_recomendacion.generar_leyes import generar


SEP = "═" * 70


def _fila(r) -> str:
    conf = {"alto": "🟢", "medio": "🟡", "bajo": "🔴"}.get(r["confianza"], "")
    return (f"   • {r['partido']:34s} → {r['apuesta']}\n"
            f"     {100*r['prob']:4.1f}%  pago x{r['pago']:<5}  confianza {conf} {r['confianza']}\n"
            f"     ↳ {r['motivo']}")


def imprimir_reporte(leyes: dict, cfg: dict | None = None) -> None:
    f = leyes["fecha"]
    titulo = nombre_competicion(cfg).upper()
    print(f"\n{SEP}")
    print(f"  ⚽  PREDICCIONES {titulo} — {f}   ({leyes['n_partidos']} partidos)")
    print(f"  (juego amistoso entre amigos — no es asesoría de apuestas reales)")
    print(SEP)

    if leyes["n_partidos"] == 0:
        print("\n  No hay partidos en esta fecha.")
        return

    print("\n🔒  LEY SEGURA  (alta probabilidad, bajo riesgo)")
    print("─" * 70)
    if leyes["segura"]:
        for r in leyes["segura"]:
            print(_fila(r))
    else:
        print("   (Ningún partido alcanza el umbral de seguridad hoy.)")

    print("\n⚖️   LEY ARRIESGADA  (probabilidad media, mejor pago)")
    print("─" * 70)
    if leyes["arriesgada"]:
        for r in leyes["arriesgada"]:
            print(_fila(r))
    else:
        print("   (Sin candidatos en la banda media hoy.)")

    print("\n🚀  LEY SOÑADOR  (baja probabilidad, alto valor)")
    print("─" * 70)
    for r in leyes["sonador"]:
        print(_fila(r))
    if leyes["parlay"]:
        p = leyes["parlay"]
        print("\n   🎯 COMBINADA SOÑADORA (parlay de las apuestas más seguras del día):")
        for a in p["apuestas"]:
            print(f"      + {a}")
        print(f"      = probabilidad {100*p['prob']:.1f}%   pago x{p['pago']}")

    print(f"\n{SEP}\n")


def imprimir_eliminatorias(con, modelo, cfg, fecha: str) -> None:
    """Probabilidades de clasificación de las eliminatorias que se juegan ese día."""
    comp = competicion_id(cfg)
    liga = liga_de_competicion(comp, cfg)
    partidos = db.calendario_de_fecha(con, fecha, comp)
    con_leg = [p for p in partidos if p["leg"]]
    if not con_leg:
        print("\n  (No hay eliminatorias a doble partido en esta fecha.)")
        return
    # La temporada sale de los propios partidos del día: una eliminatoria de
    # febrero de 2027 pertenece a la temporada 2026-27, no a la del año natural.
    ronda, temporada = con_leg[0]["ronda"], con_leg[0]["temporada"]

    # Toda la ronda, para poder emparejar ida y vuelta.
    de_la_ronda = con.execute(
        "SELECT * FROM partidos WHERE competicion=? AND ronda=? AND temporada=?",
        (comp, ronda, temporada)).fetchall()

    equipos_hoy = {p["local"] for p in partidos} | {p["visitante"] for p in partidos}
    print(f"\n{SEP}")
    print(f"  🔀  ELIMINATORIAS — {ronda} — {fecha}")
    print(SEP)
    for tie in elim.emparejar_legs(de_la_ronda):
        if not ({tie["equipo_a"], tie["equipo_b"]} & equipos_hoy):
            continue
        try:
            r = elim.eliminatoria(modelo, tie["equipo_a"], tie["equipo_b"],
                                  tie["marcador_ida"], liga)
        except KeyError as e:
            print(f"   • {tie['equipo_a']} vs {tie['equipo_b']}: sin datos ({e})")
            continue
        ida = (f"ida {r['marcador_ida'][0]}-{r['marcador_ida'][1]}"
               if r["ida_jugada"] else "ida por jugar")
        print(f"\n   • {tie['equipo_a']} vs {tie['equipo_b']}   ({ida})")
        print(f"     Pasa {tie['equipo_a']}: {100*r['pasa_a']:5.1f}%   "
              f"Pasa {tie['equipo_b']}: {100*r['pasa_b']:5.1f}%")
        print(f"     Prórroga {100*r['prorroga']:4.1f}%   "
              f"Penales {100*r['penales']:4.1f}%")
        ou = elim.over_under_global(r, (2.5, 3.5, 4.5))
        print("     Global: " + "  ".join(
            f"O{l} {100*ou[f'over_{l}']:.0f}%" for l in (2.5, 3.5, 4.5)))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Predicciones y apuestas recomendadas de la competición activa.")
    parser.add_argument("--fecha", default=date.today().isoformat(),
                        help="Fecha YYYY-MM-DD (defecto: hoy).")
    parser.add_argument("--actualizar", action="store_true",
                        help="Descargar datos frescos antes de predecir.")
    parser.add_argument("--detalle", action="store_true",
                        help="Mostrar también la tabla de mercados por partido.")
    parser.add_argument("--eliminatorias", action="store_true",
                        help="Probabilidades de clasificación de las eliminatorias del día.")
    args = parser.parse_args()

    cfg = cargar_config()
    con = db.conectar()
    db.inicializar(con)

    if args.actualizar:
        from ..fase1_datos.actualizar_diario import actualizar
        actualizar(con, cfg)

    print("Entrenando el modelo con el histórico disponible...")
    modelo = entrenar_modelo(con, cfg)

    leyes = generar(con, modelo, cfg, args.fecha)
    imprimir_reporte(leyes, cfg)

    if args.detalle:
        print("DETALLE POR PARTIDO")
        for row in db.calendario_de_fecha(con, args.fecha):
            p = predecir(con, modelo, cfg, row["local"], row["visitante"],
                         row["neutral"], competicion=row["competicion"])
            imprimir_prediccion(p)

    if args.eliminatorias:
        imprimir_eliminatorias(con, modelo, cfg, args.fecha)

    con.close()


if __name__ == "__main__":
    main()
