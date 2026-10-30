#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ "$MUJOCO_GL" == "osmesa" && -d "$project_dir/.runtime/mesa/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$project_dir/.runtime/mesa/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
runtime_python="$project_dir/.venv_mjlab/bin/python"
if [[ ! -x "$runtime_python" ]]; then
  runtime_python="$project_dir/.venv/bin/python"
fi
exec "$runtime_python" "$@"
