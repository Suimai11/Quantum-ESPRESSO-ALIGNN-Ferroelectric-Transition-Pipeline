#!/bin/bash
# ==================================================================
#  铁电相变加速计算 - 一键运行脚本 (Linux)
#  Ferroelectric transition pipeline - one script to rule them all
#
#  用法:
#    ./run_ferro.sh <输入文件> [选项]
#
#  示例:
#    ./run_ferro.sh para.in
#    ./run_ferro.sh para.POSCAR -o my_results -a 0.15
#    ./run_ferro.sh para.in --no-qe --fmax 0.02
#
#  环境变量 (可选):
#    FERRO_OUTPUT, FERRO_FMAX, FERRO_N_INTERP, FERRO_ECUTWFC,
#    FERRO_KPTS, FERRO_PSEUDO_DIR, FERRO_AMPLITUDE, etc.
# ==================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 颜色输出 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---- Banner ----
echo ""
echo "=============================================="
echo "  铁电相变加速计算流水线"
echo "  ALIGNN-FF (GPU) + Quantum ESPRESSO (GPU)"
echo "=============================================="
echo ""

# ---- Check Python ----
command -v python3 >/dev/null 2>&1 || { error "python3 not found"; exit 1; }
info "Python: $(python3 --version)"

# ---- Check CUDA ----
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device count:', torch.cuda.device_count())" 2>/dev/null || warn "PyTorch not found or CUDA issue"

# ---- Check ALIGNN ----
python3 -c "from alignn.ff.ff import AlignnAtomwiseCalculator" 2>/dev/null || {
    warn "ALIGNN not installed. Installing..."
    cd alignn-main && pip install -e . && cd "$SCRIPT_DIR" || {
        error "ALIGNN installation failed"
        exit 1
    }
    info "ALIGNN installed"
}

# ---- Check pw.x ----
QE_BIN=$(find q-e-develop -name "pw.x" 2>/dev/null | head -1)
if [ -z "$QE_BIN" ]; then
    warn "pw.x not found. Checking if compilation needed..."
    if [ -f "q-e-develop/configure" ]; then
        warn "QE not compiled. You can compile with:"
        warn "  cd q-e-develop && ./configure --with-gpu=cuda --enable-openmp && make pw"
        warn "Proceeding with ALIGNN only..."
    fi
else
    info "QE pw.x: $QE_BIN"
    export FERRO_QE_DIR="$(dirname $(dirname $QE_BIN))"
fi

# ---- Check checkpoint ----
if [ -f "checkpoint_90.pt" ]; then
    info "Finetuned checkpoint: checkpoint_90.pt"
else
    warn "checkpoint_90.pt not found, will use base ALIGNN-FF model"
fi

# ---- Check ASE ----
python3 -c "import ase" 2>/dev/null || {
    info "Installing ASE..."
    pip install ase
}

# ---- Check jarvis-tools ----
python3 -c "import jarvis" 2>/dev/null || {
    info "Installing jarvis-tools..."
    pip install jarvis-tools
}

# ---- Run pipeline ----
echo ""
info "Starting pipeline..."
echo ""

# Forward all arguments to the python CLI
python3 -m ferro_pipeline.cli "$@"

echo ""
info "Pipeline finished!"
echo ""
