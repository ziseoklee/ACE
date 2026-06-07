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
    printf "uv could not be found. Installing uv...\n"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    printf "\e[32m\u2714\e[0m uv is already installed.\n"
fi

if ! command -v uv &> /dev/null; then
    printf "\e[31m\u2718\e[0m Error: uv installation failed. Please install uv manually and rerun this script.\n"
    exit 1
fi
###################################################################################################


###################################################################################################
# 1. Update git submodules.
printf "1. Updating git submodules...\n"

git submodule update --init --recursive

printf "  Ensuring DiffSBDD .gitignore ignores Python cache directories...\n"
DIFFSBDD_GITIGNORE="$PROJECT_ROOT/src/pretrained_models/DiffSBDD/.gitignore"
touch "$DIFFSBDD_GITIGNORE"
if ! grep -qx "__pycache__/" "$DIFFSBDD_GITIGNORE"; then
    printf "__pycache__/\n" >> "$DIFFSBDD_GITIGNORE"
fi
printf "  \e[32m\u2714\e[0m DiffSBDD .gitignore is configured.\n"

# Patch DiffSBDD for newer Biopython versions without modifying the checked-in submodule.
printf "  Applying DiffSBDD Biopython compatibility patch...\n"
DIFFSBDD_LIGHTNING_MODULES="$PROJECT_ROOT/src/pretrained_models/DiffSBDD/lightning_modules.py"

if [ -f "$DIFFSBDD_LIGHTNING_MODULES" ]; then
    if grep -q "from Bio.PDB.Polypeptide import three_to_one" "$DIFFSBDD_LIGHTNING_MODULES"; then
        sed -i \
            "s/from Bio.PDB.Polypeptide import three_to_one/from Bio.Data.IUPACData import protein_letters_3to1/" \
            "$DIFFSBDD_LIGHTNING_MODULES"
    fi

    if grep -q "three_to_one(res.get_resname())" "$DIFFSBDD_LIGHTNING_MODULES"; then
        sed -i \
            "s/three_to_one(res.get_resname())/protein_letters_3to1[res.get_resname().title()]/g" \
            "$DIFFSBDD_LIGHTNING_MODULES"
    fi

    printf "  \e[32m\u2714\e[0m DiffSBDD Biopython compatibility patch applied.\n"
else
    printf "  \e[31m\u2718\e[0m DiffSBDD lightning_modules.py not found: %s\n" "$DIFFSBDD_LIGHTNING_MODULES"
    exit 1
fi

printf "  Applying DiffSBDD RDKit ligand compatibility patch...\n"
DIFFSBDD_UTILS="$PROJECT_ROOT/src/pretrained_models/DiffSBDD/utils.py"

if [ -f "$DIFFSBDD_UTILS" ]; then
    if ! grep -q "if not isinstance(ligand, str):" "$DIFFSBDD_UTILS"; then
        sed -i '/^def get_pocket_from_ligand(pdb_model, ligand, dist_cutoff=8.0):$/{
            n
            s/^$/    if not isinstance(ligand, str):/
            a\
        ligand_coords = torch.from_numpy(ligand.GetConformer().GetPositions()).float()\
        resi = None
        }' "$DIFFSBDD_UTILS"

        sed -i \
            '0,/^    if ligand\.endswith("\.sdf"):/s//    elif ligand.endswith(".sdf"):/' \
            "$DIFFSBDD_UTILS"
    fi

    printf "  \e[32m\u2714\e[0m DiffSBDD RDKit ligand compatibility patch applied.\n"
else
    printf "  \e[31m\u2718\e[0m DiffSBDD utils.py not found: %s\n" "$DIFFSBDD_UTILS"
    exit 1
fi

printf "  Applying DiffSBDD Open Babel import compatibility patch...\n"
DIFFSBDD_MOLECULE_BUILDER="$PROJECT_ROOT/src/pretrained_models/DiffSBDD/analysis/molecule_builder.py"

if [ -f "$DIFFSBDD_MOLECULE_BUILDER" ]; then
    if grep -q "^import openbabel$" "$DIFFSBDD_MOLECULE_BUILDER"; then
        sed -i \
            "s/^import openbabel$/from openbabel import openbabel/" \
            "$DIFFSBDD_MOLECULE_BUILDER"
    fi

    printf "  \e[32m\u2714\e[0m DiffSBDD Open Babel import compatibility patch applied.\n"
else
    printf "  \e[31m\u2718\e[0m DiffSBDD molecule_builder.py not found: %s\n" "$DIFFSBDD_MOLECULE_BUILDER"
    exit 1
fi

# Patch GeoDiff GIN residual connection for higher-order autograd.
printf "  Applying GeoDiff GIN autograd compatibility patch...\n"
GEODIFF_GIN_ENCODER="$PROJECT_ROOT/src/pretrained_models/GeoDiff/models/encoder/gin.py"

if [ -f "$GEODIFF_GIN_ENCODER" ]; then
    if grep -q "hidden += conv_input" "$GEODIFF_GIN_ENCODER"; then
        sed -i \
            "s/hidden += conv_input/hidden = hidden + conv_input/" \
            "$GEODIFF_GIN_ENCODER"
    fi

    printf "  \e[32m\u2714\e[0m GeoDiff GIN autograd compatibility patch applied.\n"
else
    printf "  \e[31m\u2718\e[0m GeoDiff GIN encoder not found: %s\n" "$GEODIFF_GIN_ENCODER"
    exit 1
fi

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
# 2. Sync the project with the latest changes from the main branch.
printf "2. Syncing project with the latest changes from the main branch...\n"
uv sync
###################################################################################################

printf "\e[32m\u2714\e[0m Project setup complete!\n"