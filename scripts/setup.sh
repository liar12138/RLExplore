#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:---reuse-server-runtimes}"
if [[ "$mode" == "--reuse-server-runtimes" ]]; then
  mjlab_python=/home/tcs/1t01/envs/capylab/bin/python
  capylab_packages=/home/tcs/1t01/envs/capylab/lib/python3.10/site-packages
  [[ -x "$mjlab_python" && -d "$capylab_packages" ]]
  "$mjlab_python" -m venv "$project_dir/.venv_mjlab"
  printf '%s\n' "$capylab_packages" > "$project_dir/.venv_mjlab/lib/python3.10/site-packages/mjlab_runtime.pth"
elif [[ "$mode" == "--standard" ]]; then
  "${PYTHON_BIN:-python3}" -m venv "$project_dir/.venv_mjlab"
  "$project_dir/.venv_mjlab/bin/python" -m pip install -e "$project_dir"
else
  printf '%s\n' 'Usage: scripts/setup.sh [--reuse-server-runtimes|--standard]' >&2
  exit 2
fi
printf '%s\n' 'Project runtime ready. Run ./scripts/run.sh doctor --set device=cuda'
