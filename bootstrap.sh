#!/usr/bin/env bash
set -euo pipefail

# Make sure uv is available
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH")

# Private Python 3.11 env (seed pip)
uv venv --python 3.11 --seed .venv
. .venv/bin/activate

# Basic build tools
uv pip install -U pip wheel

# Dependencies (prefer lock)
if [ -f requirements.lock ]; then
  uv pip sync requirements.lock
else
  uv pip install -r code/requirements.txt
fi

# Run full pipeline + verify
chmod +x code/run_si.sh
bash code/run_si.sh 2020 2021 2022 2023
bash code/run_si.sh verify
