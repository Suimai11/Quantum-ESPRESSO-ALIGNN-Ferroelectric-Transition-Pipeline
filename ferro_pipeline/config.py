"""
Configuration - minimal, most params are read from the QE input file.
"""

import os
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # Paths
    work_dir: str = "."
    output_dir: str = "ferro_results"
    checkpoint_path: str = "checkpoint_90.pt"
    qe_source_dir: str = "q-e-develop"

    # ALIGNN-FF
    alignn_device: str = "cuda"
    alignn_model_path: str = ""
    fmax_alignn: float = 0.05
    max_steps_alignn: int = 300
    relax_cell_alignn: bool = True

    # Interpolation (only these two are pipeline-specific)
    n_interp_images: int = 9
    distortion_amplitude: float = 0.1

    # QE params will be READ FROM INPUT FILE, these are fallbacks
    pseudo_dir: str = ""
    qe_ecutwfc: float = 60.0
    qe_ecutrho: float = 480.0
    qe_kpts: list = field(default_factory=lambda: [4, 4, 4])
    qe_scf_thr: float = 1e-8

    # QE execution
    use_qe: bool = True
    qe_npool: int = 1

    # Output
    save_intermediate: bool = True
    verbose: bool = True

    @classmethod
    def from_qe_input(cls, input_path: str) -> "Config":
        """Read QE input file and extract all relevant parameters."""
        c = cls()
        c._input_path = input_path
        return c

    def to_dict(self) -> dict:
        return asdict(self)
