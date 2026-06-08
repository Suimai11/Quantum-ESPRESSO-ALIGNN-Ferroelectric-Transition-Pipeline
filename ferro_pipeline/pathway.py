"""
Ferroelectric pathway generation.
"""

import numpy as np
from typing import List


def add_ferroelectric_distortion(atoms, amplitude: float = 0.1):
    """
    Apply ferroelectric distortion to paraelectric structure.

    Generates two opposite polar distortions (initial and final states).

    Args:
        atoms: ASE Atoms (paraelectric phase)
        amplitude: distortion amplitude in Angstrom

    Returns:
        (atoms_initial, atoms_final)
    """
    from ase import Atoms

    a = atoms.copy()
    pos = a.get_positions()
    syms = a.get_chemical_symbols()
    cell = a.get_cell()

    z = pos[:, 2]
    z_min, z_max = z.min(), z.max()
    dz = z_max - z_min if z_max > z_min else 1.0

    # Sinusoidal displacement along c-axis
    disp = amplitude * np.sin(np.pi * (z - z_min) / dz)

    pos_i = pos.copy()
    pos_f = pos.copy()
    pos_i[:, 2] += disp
    pos_f[:, 2] -= disp

    atoms_i = Atoms(symbols=syms, positions=pos_i, cell=cell, pbc=True)
    atoms_f = Atoms(symbols=syms, positions=pos_f, cell=cell, pbc=True)
    return atoms_i, atoms_f


def linear_interpolate(atoms_i, atoms_f, n_images: int = 9) -> List:
    from ase import Atoms

    pos_i = atoms_i.get_positions()
    pos_f = atoms_f.get_positions()
    cell_i = atoms_i.get_cell().array
    cell_f = atoms_f.get_cell().array
    syms = atoms_i.get_chemical_symbols()

    images = []
    for idx in range(n_images):
        t = idx / max(n_images - 1, 1)
        im = Atoms(symbols=syms,
                   positions=(1 - t) * pos_i + t * pos_f,
                   cell=(1 - t) * cell_i + t * cell_f,
                   pbc=True)
        images.append(im)
    return images
