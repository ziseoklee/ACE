#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)/molecule"

NPROC="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"


###################################################################################################
# 0. Check if `uv` is installed. If not, install it.
printf "0. Checking if uv is installed...\n"

if ! command -v uv &> /dev/null; then
    printf "  uv could not be found. Installing uv...\n"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    printf "  \e[32m\u2714\e[0m uv is already installed.\n"
fi

if ! command -v uv &> /dev/null; then
    printf "  \e[31m\u2718\e[0m Error: uv installation failed. Please install uv manually and rerun this script.\n"
    exit 1
fi

PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
    printf "  \e[31m\u2718\e[0m Error: python could not be found. Please install Python and rerun this script.\n"
    exit 1
fi
###################################################################################################


###################################################################################################
# 1. Update git submodules.
printf "1. Updating git submodules...\n"

git submodule update --init --recursive

printf "  Applying pretrained runtime compatibility patches...\n"
"$PYTHON_BIN" "$PROJECT_ROOT/assets/pretrained_packaging/runtime_fixes.py"

printf "  Applying pretrained dependency packaging patches...\n"
"$PYTHON_BIN" "$PROJECT_ROOT/assets/pretrained_packaging/dependency_resolve.py"

# Download DiffSBDD checkpoints
printf "  Downloading DiffSBDD checkpoints...\n"
pushd "$PROJECT_ROOT"/src/pretrained_models/DiffSBDD > /dev/null

if [ -f "checkpoints/crossdocked_ca_cond.ckpt" ] && [ -f "checkpoints/crossdocked_fullatom_cond.ckpt" ]; then
    printf "  \e[32m\u2714\e[0m DiffSBDD checkpoints already exist. Skipping download.\n"
else
    printf "  \e[31m\u2718\e[0m DiffSBDD checkpoints not found. Downloading...\n"
    mkdir -p checkpoints
    wget -P checkpoints/ https://zenodo.org/record/8183747/files/crossdocked_ca_cond.ckpt
    wget -P checkpoints/ https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt
fi
popd > /dev/null

# Download GeoDiff checkpoints
printf "  Downloading GeoDiff checkpoints...\n"
pushd "$PROJECT_ROOT"/src/pretrained_models/GeoDiff > /dev/null

if [ -f "log/model/checkpoints/drugs_default.pt" ] && [ -f "log/model/checkpoints/qm9_default.pt" ]; then
    printf "  \e[32m\u2714\e[0m GeoDiff checkpoints already exist. Skipping download.\n"
else
    printf "  \e[31m\u2718\e[0m GeoDiff checkpoints not found. Downloading...\n"
    mkdir -p log/model
    pushd log/model > /dev/null
    # GeoDiff checkpoints from https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh
    uvx gdown 1zylCnk3CLylwg_47yWxczCfhfbF1Es6b
    tar -xzf checkpoints.tar.gz
    rm checkpoints.tar.gz
    popd > /dev/null
    # Copy model config yaml files
    cp configs/drugs_1k_default.yml log/model/
    cp configs/qm9_default.yml log/model/
fi
popd > /dev/null
###################################################################################################


###################################################################################################
# 2. Create/update this experiment's local .venv from the committed lockfile.
printf "2. Syncing the molecule environment from uv.lock...\n"
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    uv venv --python 3.11
fi
uv sync --frozen
###################################################################################################

printf "\e[32m\u2714\e[0m Project setup complete!\n"
