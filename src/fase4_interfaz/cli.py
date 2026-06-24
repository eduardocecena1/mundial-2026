"""
cli.py — FASE 4: interfaz de línea de comandos.

Imprime un reporte limpio y ordenado de las predicciones y las 3 Leyes para una
fecha del Mundial 2026.

Uso (desde la raíz del proyecto):
    python predicciones.py --fecha 2026-06-25
    python predicciones.py --fecha 2026-06-25 --actualizar   # baja datos frescos antes
    python predicciones.py --fecha 2026-06-25 --detalle       # + tabla por partido
"""

from __future__ import annotations

import argparse
from datetime import date

from ..fase1_datos import db
from ..fase2_modelo.entrenar import cargar_config, entrenar_modelo
from ..fase2_modelo.predecir_partido import predecir, imprimir_prediccion
from ..fase3_recomendacion.generar_leyes import generar


SEP = "═" * 70


def _fila(r) -> str:
    conf = {"alto": "🟢", "medio": "🟡", "bajo": "🔴"}.get(r["confianza"], "")
    return (f"   • {r['partido']:34s} → {r['apuesta']}\n"
            f"     {100*r['prob']:4.1f}%  pago x{r['pago']:<5}  confianza {conf} {r['confianza']}\n"
            f"     ↳ {r['motivo']}")


def imprimir_reporte(leyes: dict) -> None:
    f = leyes["fecha"]
    print(f"\n{SEP}")
    print(f"  ⚽  PREDICCIONES MUNDIAL 2026 — {f}   ({leyes['n_partidos']} partidos)")
    print(f"  (juego amistoso entre amigos — no es asesoría de apuestas reales)")
    print(SEP)

    if leyes["n_partidos"] == 0:
        print("\n  No hay partidos del Mundial 2026 en esta fecha.")
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


def main():
    parser = argparse.ArgumentParser(
        description="Predicciones y apuestas recomendadas del Mundial 2026.")
    parser.add_argument("--fecha", default=date.today().isoformat(),
                        help="Fecha YYYY-MM-DD (defecto: hoy).")
    parser.add_argument("--actualizar", action="store_true",
                        help="Descargar datos frescos antes de predecir.")
    parser.add_argument("--detalle", action="store_true",
                        help="Mostrar también la tabla de mercados por partido.")
    args = parser.parse_args()

    con = db.conectar()
    db.inicializar(con)

    if args.actualizar:
        from ..fase1_datos.actualizar_diario import actualizar
        actualizar(con)

    cfg = cargar_config()
    print("Entrenando el modelo con el histórico disponible...")
    modelo = entrenar_modelo(con, cfg)

    leyes = generar(con, modelo, cfg, args.fecha)
    imprimir_reporte(leyes)

    if args.detalle:
        print("DETALLE POR PARTIDO")
        for row in db.calendario_de_fecha(con, args.fecha):
            p = predecir(con, modelo, cfg, row["local"], row["visitante"], row["neutral"])
            imprimir_prediccion(p)

    con.close()


if __name__ == "__main__":
    main()
