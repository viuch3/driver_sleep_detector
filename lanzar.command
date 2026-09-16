  #!/bin/bash
cd "$(dirname "$0")" || exit 1
for CANDIDATO in ".venv/bin/python" "backend/.venv/bin/python"; do
    [ -x "$CANDIDATO" ] && PY="$CANDIDATO" && break
done
[ -z "$PY" ] && echo "No se encontro el entorno virtual" && read -r && exit 1
"$PY" app.py
