#!/usr/bin/env python3
"""
predicciones.py — Punto de entrada del proyecto.

Atajo para correr la interfaz de línea de comandos:

    python predicciones.py --fecha 2026-06-25
    python predicciones.py --fecha 2026-06-25 --actualizar
    python predicciones.py --detalle

Es para un juego amistoso entre amigos: el enfoque es diversión y presumir
aciertos, NO asesoría financiera ni apuestas reales con dinero.
"""

from src.fase4_interfaz.cli import main

if __name__ == "__main__":
    main()
