#!/usr/bin/env bash
set -euo pipefail

EXTENSIONS=(
  github.copilot
  github.copilot-chat
  eamodio.gitlens
  github.vscode-github-actions
  ms-python.python
  ms-python.vscode-pylance
  charliermarsh.ruff
  batisteo.vscode-django
  mikestead.dotenv
  humao.rest-client
  dbaeumer.vscode-eslint
  esbenp.prettier-vscode
  ecmel.vscode-html-css
  redhat.vscode-yaml
  ms-azuretools.vscode-docker
  ms-azuretools.vscode-azureappservice
  ms-azuretools.vscode-azurestaticwebapps
  vscjava.vscode-java-pack
  vscjava.vscode-gradle
  yzhang.markdown-all-in-one
  streetsidesoftware.code-spell-checker
)

CODE_CLI=""

pick_remote_cli() {
  # Cloud Shell / VS Code Server: suele existir un CLI funcional aquí.
  # Tomamos el primero que encontremos.
  find /home/juan/code/servers -type f -path '*/server/bin/remote-cli/code' 2>/dev/null | head -n 1
}

if command -v code >/dev/null 2>&1; then
  # En algunos entornos (Cloud Shell), /usr/bin/code es un wrapper que falla y muestra HTML.
  if out=$(code --version 2>/dev/null); then
    if echo "$out" | grep -qi '<!DOCTYPE html>'; then
      CODE_CLI=""
    else
      CODE_CLI="code"
    fi
  fi
fi

if [ -z "${CODE_CLI}" ]; then
  remote_cli="$(pick_remote_cli)"
  if [ -n "${remote_cli}" ] && [ -x "${remote_cli}" ]; then
    CODE_CLI="${remote_cli}"
  fi
fi

if [ -z "${CODE_CLI}" ]; then
  echo "No encuentro un CLI funcional de VS Code para instalar extensiones." >&2
  echo "- Intenta instalar manualmente desde Extensions (UI) o asegura que exista el CLI remoto en /home/juan/code/servers/..." >&2
  exit 1
fi

echo "Instalando extensiones recomendadas (VS Code)..."
for ext in "${EXTENSIONS[@]}"; do
  echo "- $ext"
  "${CODE_CLI}" --install-extension "$ext" --force >/dev/null
done

echo "OK. Si VS Code estaba abierto, puede requerir Reload/Restart para activar todo." 
