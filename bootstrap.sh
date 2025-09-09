#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH")

uv venv --python 3.11 --seed .venv
. .venv/bin/activate
uv pip install -U pip wheel
(uv pip sync requirements.lock || uv pip install -r code/requirements.txt)

chmod +x code/run_si.sh
bash code/run_si.sh 2020 2021 2022 2023
bash code/run_si.sh verify
