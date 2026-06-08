# Ferroelectric Transition Pipeline

**铁电相变加速计算流水线**

ALIGNN-FF (GPU) 快速弛豫 + Quantum ESPRESSO (GPU) 高精度精算，一键输出铁电相变能垒数据。

---

## 目录

- [环境配置](#环境配置)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [示例：Ag₂S 铁电相变](#示例ag₂s-铁电相变)
- [完整用法](#完整用法)
- [输出说明](#输出说明)

---

## 环境配置

### 1. Python 环境

```bash
# 推荐 Python 3.10
sudo apt update
sudo apt install -y python3 python3-pip
```

### 2. NVIDIA GPU 驱动 + CUDA

```bash
# 查看 GPU
nvidia-smi

# CUDA 版本应 ≥ 11.8
nvcc --version
```

### 3. PyTorch (CUDA 版)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

验证：
```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
# 输出应为 CUDA: True
```

### 4. DGL (CUDA 版)

```bash
pip install dgl -f https://data.dgl.ai/wheels/cu124/repo.html
```

### 5. ALIGNN-FF

```bash
# 进入项目后
cd alignn-main && pip install -e . && cd ..
```

### 6. Quantum ESPRESSO (GPU 版)

项目内已包含源码 `qe_src/`，按以下步骤编译：

```bash
# 安装编译依赖
sudo apt install -y gcc gfortran cmake openmpi-bin libopenmpi-dev

# 编译 GPU 版 QE（需要 NVHPC 编译器）
cd qe_src/qe_src
mkdir build && cd build

cmake .. \
  -DCMAKE_C_COMPILER=nvc \
  -DCMAKE_Fortran_COMPILER=nvfortran \
  -DQE_ENABLE_CUDA=ON \
  -DQE_ENABLE_OPENMP=ON

make -j$(nproc)

# 验证
ls bin/pw.x
```

> **注意**：如果在 Windows WSL2 下，NVHPC 编译器需要单独安装。  
> 暂时无法编译 QE 时，可以使用 `--no-qe` 只运行 ALIGNN-FF 部分。

### 7. 其他 Python 依赖

```bash
pip install ase jarvis-tools pandas numpy matplotlib scipy
```

---

## 项目结构

```
QE_IF/
├── qe_src/                  ← Quantum ESPRESSO 源码（GPU 版）
│   └── qe_src/build/bin/pw.x  ← 编译后的 pw.x
├── alignn-main/             ← ALIGNN-FF 源码
├── checkpoint_90.pt         ← ALIGNN-FF 微调权重
├── ferro_pipeline/          ← 流水线主代码
│   ├── __init__.py
│   ├── config.py
│   ├── io_utils.py          ← 读取 QE 输入文件 + 结构
│   ├── alignn_engine.py     ← ALIGNN-FF GPU 引擎
│   ├── qe_engine.py         ← QE GPU 引擎
│   ├── pathway.py           ← 铁电畸变 + 路径插值
│   ├── pipeline.py          ← 主流程
│   └── cli.py               ← 命令行入口
├── run_ferro.sh             ← 一键运行脚本
└── README.md
```

---

## 快速开始

**把 QE 输入文件和赝势放一个文件夹，一行命令跑：**

```bash
# 文件夹结构示例：
# my_calc/
# ├── para.in           ← QE 输入文件（或 .cif / POSCAR）
# ├── Ag.UPF            ← 赝势
# └── S.UPF

python3 -m ferro_pipeline.cli my_calc/
```

程序会自动：
1. 读取 QE 输入文件中的参数（`pseudo_dir`、`ecutwfc`、`kpts` 等）
2. 在 `pseudo_dir` 中自动找到对应元素的 `.UPF` 赝势文件
3. 构架铁电畸变 → ALIGNN-FF 弛豫 → 路径扫描 → QE 精算 → 输出 CSV

---

## 示例：Ag₂S 铁电相变

### 准备输入

`Ag2S.in` 放在 `/mnt/f/compute/` 下，和赝势文件在一起：

```
compute/
├── Ag2S.in
├── Ag_ONCV_PBE-1.0.oncvpsp.upf
└── s_pbe_v1.4.uspp.F.UPF
```

`Ag2S.in` 内容：

```
&CONTROL
  calculation = 'scf'
  prefix = 'Ag2S'
  pseudo_dir = './'
  outdir = './out'
  tprnfor = .true.
  tstress = .true.
/
&SYSTEM
  ibrav = 0
  nat = 12
  ntyp = 2
  ecutwfc = 60
  ecutrho = 480
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1e-8
/
ATOMIC_SPECIES
Ag  107.8682  Ag_ONCV_PBE-1.0.oncvpsp.upf
S   32.065    s_pbe_v1.4.uspp.F.UPF

CELL_PARAMETERS (angstrom)
  4.81887000  0.00000000  0.00000000
  0.00000000  6.96885800  0.00000000
  0.00000000  0.00000000  7.57733700

ATOMIC_POSITIONS (crystal)
Ag  0.00000000  0.12010800  0.25000000
Ag  0.00000000  0.50000000  0.00000000
S   0.00000000  0.27971000  0.75000000
Ag  0.50000000  0.62010800  0.75000000
Ag  0.50000000  0.00000000  0.50000000
S   0.50000000  0.77971000  0.25000000
Ag  0.00000000  0.87989200  0.75000000
Ag  0.00000000  0.50000000  0.50000000
S   0.00000000  0.72029000  0.25000000
Ag  0.50000000  0.37989200  0.25000000
Ag  0.50000000  0.00000000  0.00000000
S   0.50000000  0.22029000  0.75000000

K_POINTS (automatic)
4 4 4 0 0 0
```

### 运行

```bash
# 完整流程（ALIGNN + QE GPU）
cd QE_IF
python3 -m ferro_pipeline.cli /mnt/f/compute/ -o /mnt/f/compute/results

# 仅 ALIGNN（跳过 QE，几秒出结果）
python3 -m ferro_pipeline.cli /mnt/f/compute/ --no-qe -o /mnt/f/compute/results
```

### 输出

```
compute/results/
├── ferro_path_data.csv    ← 能垒数据（9 个插值点）
├── relaxed_initial.POSCAR
├── relaxed_final.POSCAR
├── qe_refined.csv         ← QE 精算数据（开启 QE 时）
└── report.json
```

---

## 完整用法

```bash
python3 -m ferro_pipeline.cli <输入> [选项]
```

### 输入

| 方式 | 说明 | 示例 |
|------|------|------|
| **文件夹** | 自动找到里面的 `.in` 或 `.cif` 文件 | `python3 -m ferro_pipeline.cli my_calc/` |
| **QE 输入文件** | 直接指定 `.in` 文件 | `python3 -m ferro_pipeline.cli my_calc/para.in` |
| **CIF 文件** | 直接读取结构 | `python3 -m ferro_pipeline.cli structure.cif` |

### 选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-o DIR` | 输出目录 | `ferro_results/` |
| `-a AMP` | 畸变幅度 (Å) | `0.1` |
| `-n NUM` | 插值图像数 | `9` |
| `--fmax VAL` | ALIGNN 力收敛标准 (eV/Å) | `0.05` |
| `--no-qe` | 跳过 QE 精算 | 不跳过 |
| `--checkpoint PATH` | ALIGNN 权重文件 | `checkpoint_90.pt` |

---

## 输出说明

### ferro_path_data.csv

| 列 | 说明 | 单位 |
|----|------|------|
| `image` | 插值点索引 | — |
| `t` | 反应坐标 (0→1) | — |
| `energy_eV` | 总能 | eV |
| `energy_per_atom_eV` | 每原子能量 | eV/atom |
| `fmax_eV_per_Ang` | 最大原子力 | eV/Å |
| `volume_Ang3` | 晶胞体积 | Å³ |
| `a_Ang`, `b_Ang`, `c_Ang` | 晶格参数 | Å |

### 步骤说明

```
 输入: 顺电相结构（.in / .cif / POSCAR）
           │
           ▼
  [1] 构建铁电畸变（初态 + 末态）
           │
           ▼
  [2] ALIGNN-FF 粗弛豫（GPU）
           │
           ▼
  [3] 路径插值 + 能量扫描（ALIGNN-FF）
           │
           ▼
  [4] QE GPU 精算（关键点）
           │
           ▼
 输出: ferro_path_data.csv
```
使用前要从https://huggingface.co/Suimai/alignn-ferroelectric-phase-transition/resolve/main/checkpoint_90.pt?download=true下载模型并放到文件夹最上层，并要下载alignn源码与qe（gpu版）源码
