"""
Structure I/O and QE input parsing.
Reads structure AND QE parameters (ecutwfc, kpts, pseudo_dir, etc.) from input file.
"""

import re
import os
import numpy as np
from ase.io import read, write
from ase import Atoms

# ---- Read structure ----

def read_structure(path: str) -> Atoms:
    """
    Read a crystal structure from any ASE-supported format.
    For QE input (.in), uses ASE's 'espresso-in' reader.
    """
    try:
        return read(path)
    except Exception:
        if path.endswith(".in"):
            return read(path, format="espresso-in")
        raise


def write_poscar(atoms: Atoms, path: str):
    write(path, atoms, format="vasp")


# ---- Parse QE input parameters ----

def parse_qe_input(path: str) -> dict:
    """
    Parse a QE input file and extract key parameters.

    Returns dict with keys:
        pseudo_dir, ecutwfc, ecutrho, kpts, conv_thr,
        atomic_species (list of (name, mass, pseudopotential)),
        calculation, prefix, outdir
    """
    params = {}
    with open(path) as f:
        text = f.read()

    # Remove comments
    text = re.sub(r"!\s*[^\n]*", "", text)

    # Helper: find value of a key in a namelist
    def find_in(line, key):
        """Extract value for key from a line like 'key = value'"""
        m = re.search(rf"\b{re.escape(key)}\s*=\s*([^,!\n]+)", line, re.I)
        if m:
            return m.group(1).strip().strip("'").strip('"')
        return None

    # Split into sections: &CONTROL ... /, &SYSTEM ... /, etc.
    sections = {}
    current_section = None
    buffer = []
    for line in text.split("\n"):
        stripped = line.strip()
        m_section = re.match(r"^\s*&([A-Z_]+)", stripped, re.I)
        if m_section:
            current_section = m_section.group(1).upper()
            buffer = [line]
            continue
        if stripped == "/" and current_section:
            buffer.append(line)
            sections[current_section] = "\n".join(buffer)
            current_section = None
            buffer = []
            continue
        if current_section:
            buffer.append(line)

    # --- &CONTROL ---
    control = sections.get("CONTROL", "")
    if control:
        for k in ("pseudo_dir", "outdir", "prefix", "calculation"):
            v = find_in(control, k)
            if v:
                params[k] = v

    # --- &SYSTEM ---
    system = sections.get("SYSTEM", "")
    if system:
        for k in ("ecutwfc", "ecutrho", "nat", "ntyp", "nbnd", "occupations",
                   "smearing", "degauss", "input_dft", " Hubbard_U"):
            v = find_in(system, k)
            if v:
                try:
                    params[k] = float(v) if "." in v else int(v)
                except ValueError:
                    params[k] = v

    # --- &ELECTRONS ---
    electrons = sections.get("ELECTRONS", "")
    if electrons:
        v = find_in(electrons, "conv_thr")
        if v:
            params["conv_thr"] = float(v)

    # --- K_POINTS ---
    m_k = re.search(r"K_POINTS\s*(?:\{[^}]*\})?\s*\n\s*(\d+)\s+(\d+)\s+(\d+)", text, re.I)
    if m_k:
        params["kpts"] = [int(m_k.group(1)), int(m_k.group(2)), int(m_k.group(3))]

    # --- ATOMIC_SPECIES ---
    in_species = False
    species_list = []
    for line in text.split("\n"):
        s = line.strip()
        if "ATOMIC_SPECIES" in s:
            in_species = True
            continue
        if in_species and (s.startswith("ATOMIC_POSITIONS") or s.startswith("CELL_PARAMETERS")):
            in_species = False
        if in_species and s and not s.startswith("!"):
            parts = s.split()
            if len(parts) >= 3:
                species_list.append({
                    "name": parts[0],
                    "mass": parts[1],
                    "pseudo": parts[2],
                })
    if species_list:
        params["atomic_species"] = species_list

    # Resolve pseudo_dir: if it's a relative path, make it absolute relative to input file
    if "pseudo_dir" in params:
        pd = params["pseudo_dir"]
        if not os.path.isabs(pd):
            # relative to input file's directory
            base = os.path.dirname(os.path.abspath(path))
            resolved = os.path.join(base, pd)
            if os.path.isdir(resolved):
                params["pseudo_dir"] = resolved
            else:
                # also try relative to CWD
                if os.path.isdir(os.path.abspath(pd)):
                    params["pseudo_dir"] = os.path.abspath(pd)

    return params


def write_qe_input(atoms: Atoms, params: dict, job_name: str = "scf",
                   calculation: str = "scf") -> str:
    """
    Generate QE input file string from atoms + parameters.
    Uses the same parameters extracted from the original input file.
    """
    cell = atoms.get_cell()
    pos = atoms.get_positions()
    sym = atoms.get_chemical_symbols()
    spec = sorted(set(sym), key=lambda x: sym.index(x))
    nat, ntyp = len(sym), len(spec)

    pseudo_dir = params.get("pseudo_dir", ".")
    ecutwfc = params.get("ecutwfc", 60.0)
    ecutrho = params.get("ecutrho", 480.0)
    kpts = params.get("kpts", [4, 4, 4])
    conv_thr = params.get("conv_thr", 1e-8)
    prefix = params.get("prefix", job_name)
    outdir = params.get("outdir", f"./qe_{job_name}_out")

    lines = []
    lines.append("&CONTROL")
    lines.append(f"  calculation = '{calculation}'")
    lines.append(f"  prefix = '{prefix}'")
    lines.append(f"  pseudo_dir = '{pseudo_dir}'")
    lines.append(f"  outdir = '{outdir}'")
    lines.append(f"  tprnfor = .true.")
    lines.append(f"  tstress = .true.")
    lines.append("/")
    lines.append("&SYSTEM")
    lines.append(f"  ibrav = 0")
    lines.append(f"  nat = {nat}")
    lines.append(f"  ntyp = {ntyp}")
    lines.append(f"  ecutwfc = {ecutwfc}")
    lines.append(f"  ecutrho = {ecutrho}")
    lines.append(f"  occupations = 'smearing'")
    lines.append(f"  smearing = 'gaussian'")
    lines.append(f"  degauss = 0.01")
    lines.append("/")
    lines.append("&ELECTRONS")
    lines.append(f"  conv_thr = {conv_thr}")
    lines.append(f"  mixing_beta = 0.7")
    lines.append("/")
    if calculation == "relax" or calculation == "vc-relax":
        lines.append("&IONS")
        lines.append("  ion_dynamics = 'bfgs'")
        lines.append("/")
    if calculation == "vc-relax":
        lines.append("&CELL")
        lines.append("  cell_dynamics = 'bfgs'")
        lines.append("/")
    lines.append("ATOMIC_SPECIES")
    # Auto-detect pseudopotential files in pseudo_dir
    species_info = params.get("atomic_species", [])
    pp_dir = params.get("pseudo_dir", ".")
    for s in spec:
        mass = 1.0
        # First check original input for mass
        for sp in species_info:
            if sp["name"].lower() == s.lower():
                mass = sp["mass"]
                break
        # Auto-find the UPF file matching the element
        pp_file = None
        if os.path.isdir(pp_dir):
            for fname in os.listdir(pp_dir):
                if fname.lower().endswith(('.upf', '.upf.gz')):
                    # Match by element name at start of filename
                    if fname.lower().startswith(s.lower()):
                        pp_file = fname
                        break
        if not pp_file:
            pp_file = f"{s.lower()}.upf"  # fallback
        lines.append(f"  {s}  {mass}  {pp_file}")
    lines.append("")
    lines.append("CELL_PARAMETERS (angstrom)")
    for i in range(3):
        lines.append(f"  {cell[i,0]:15.8f}  {cell[i,1]:15.8f}  {cell[i,2]:15.8f}")
    lines.append("")
    lines.append("ATOMIC_POSITIONS (angstrom)")
    for i in range(nat):
        lines.append(f"  {sym[i]:4s}  {pos[i,0]:15.8f}  {pos[i,1]:15.8f}  {pos[i,2]:15.8f}")
    lines.append("")
    lines.append("K_POINTS (automatic)")
    lines.append(f"  {kpts[0]} {kpts[1]} {kpts[2]}  0 0 0")
    return "\n".join(lines)
