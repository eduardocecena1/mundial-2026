# 🌐 Publicar la web (para usarla en el celular y compartirla)

La web se despliega gratis en **Streamlit Community Cloud**. Te da una URL pública
(ej. `https://tu-app.streamlit.app`) que abres en el móvil y compartes con tus amigos.

> ✅ **Se actualiza sola cada día.** En la nube, la app reconstruye los datos frescos
> desde la fuente pública (que se actualiza a diario) la primera vez que alguien la
> abre cada día. No tienes que hacer nada: abres la web el día del partido y ya están
> los resultados de ayer y los partidos de hoy.

---

## Pasos (10 minutos, una sola vez)

### 1. Crear una cuenta de GitHub (si no tienes)
Ve a <https://github.com> y regístrate (gratis).

### 2. Subir el proyecto a GitHub

**Opción A — con la web de GitHub (sin instalar nada):**
1. En GitHub, pulsa **New repository**, ponle nombre (ej. `mundial-2026`), déjalo
   **Private** si quieres, y crea.
2. GitHub te muestra unos comandos. En la terminal, dentro de la carpeta del proyecto:
   ```bash
   git remote add origin https://github.com/TU_USUARIO/mundial-2026.git
   git branch -M main
   git push -u origin main
   ```
   (Te pedirá usuario y un *token* de acceso; GitHub te guía para crearlo.)

**Opción B — con la CLI de GitHub (más fácil de empujar):**
```bash
brew install gh        # instala la CLI
gh auth login          # inicia sesión (sigue el asistente)
gh repo create mundial-2026 --private --source=. --push
```

### 3. Desplegar en Streamlit Cloud
1. Entra a <https://share.streamlit.io> y **Sign in with GitHub**.
2. Pulsa **Create app** → **Deploy a public app from GitHub**.
3. Rellena:
   - **Repository:** `TU_USUARIO/mundial-2026`
   - **Branch:** `main`
   - **Main file path:** `src/fase4_interfaz/app_streamlit.py`
4. Pulsa **Deploy**. La primera vez tarda ~1–2 min (instala dependencias y baja datos).

¡Listo! Te queda una URL tipo `https://mundial-2026-xxxx.streamlit.app`.
Ábrela en el móvil, mándala al grupo y a presumir aciertos. 🏆

---

## Notas

- **No subas el archivo `.env`** (tu key de API-Football). Ya está en `.gitignore`,
  así que no se sube. La web no necesita esa key para funcionar.
- **La base de datos no se sube** (también está ignorada): la app la reconstruye en
  la nube con datos frescos. Por eso se mantiene actualizada sola.
- Si Streamlit Cloud "duerme" la app por inactividad, se reactiva sola al abrir la
  URL (tarda unos segundos la primera vez).
- ¿Cambiaste código? `git push` y Streamlit Cloud vuelve a desplegar solo.
