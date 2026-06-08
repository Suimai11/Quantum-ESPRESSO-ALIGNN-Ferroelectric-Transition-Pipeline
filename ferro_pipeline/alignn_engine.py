"""
ALIGNN-FF engine (GPU).
- Load finetuned checkpoint
- Fast structure relaxation
- Single-point energy/forces
"""

import os, sys, warnings
import numpy as np
import torch
from typing import Tuple, Optional
from .config import Config

warnings.filterwarnings("ignore")


class AlignNNFEEngine:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.calc = None
        self._setup()

    def _setup(self):
        alignn_path = os.path.join(self.config.work_dir, "alignn-main")
        if alignn_path not in sys.path:
            sys.path.insert(0, alignn_path)

        from alignn.models.alignn import ALIGNN, ALIGNNConfig
        from alignn.ff.ff import AlignnAtomwiseCalculator

        if not os.path.exists(self.config.checkpoint_path):
            raise FileNotFoundError(
                f"[ALIGNN] Checkpoint not found: {self.config.checkpoint_path}")

        if self.config.verbose:
            print(f"[ALIGNN] Loading model from: {self.config.checkpoint_path}")

        ckpt = torch.load(self.config.checkpoint_path,
                          map_location=self.device, weights_only=False)

        # Get state_dict and config
        state = ckpt if isinstance(ckpt, dict) and "state_dict" not in ckpt else \
                ckpt.get("state_dict", ckpt.get("model_state", ckpt.get("model", ckpt)))

        config_dict = ckpt.get("config", {})
        config_dict["name"] = "alignn"

        # Build model from config, load weights
        model = ALIGNN(ALIGNNConfig(**config_dict))
        try:
            model.load_state_dict(state)
        except Exception:
            # Try nested key
            for key in ("model_state", "state_dict", "model"):
                if key in ckpt:
                    try:
                        model.load_state_dict(ckpt[key])
                        break
                    except Exception:
                        continue
        model.to(self.device)
        model.eval()

        calc_config = {
            "model": {
                "name": "alignn",
                "stresswise_weight": 0.3,
                "alignn_layers": 4,
                "gcn_layers": 4,
                "atom_input_features": 92,
                "edge_input_features": 80,
                "triplet_input_features": 40,
                "embedding_features": 64,
                "hidden_features": 256,
                "output_features": 1,
                "link": "identity",
                "zero_inflated": False,
                "classification": False,
                "num_classes": 2,
                "extra_features": 0,
            },
            "neighbor_strategy": "radius_graph",
            "use_canonize": False,
            "atom_features": "cgcnn",
            "cutoff": 8.0,
            "cutoff_radius": 8.0,
            "max_neighbors": 12,
            "num_neighbors": 12,
            "batch_size": 1,
            "num_radial": 6,
            "num_angular": 6,
        }
        self.calc = AlignnAtomwiseCalculator(
            model=model, config=calc_config,
            device=self.device, include_stress=True)

        print(f"[ALIGNN] Ready (device={self.device})")

    def get_energy(self, atoms) -> float:
        atoms.calc = self.calc
        return float(atoms.get_potential_energy())

    def get_energy_forces(self, atoms) -> Tuple[float, np.ndarray]:
        atoms.calc = self.calc
        e = float(atoms.get_potential_energy())
        f = atoms.get_forces().copy()
        return e, f

    def relax(self, atoms, fmax=None, max_steps=None):
        from ase.optimize import FIRE
        try:
            from ase.constraints import ExpCellFilter
        except ImportError:
            from ase.constraints import UnitCellFilter as ExpCellFilter

        fmax = fmax or self.config.fmax_alignn
        max_steps = max_steps or self.config.max_steps_alignn

        at = atoms.copy()
        at.calc = self.calc

        # Note: skip ExpCellFilter due to ASE 3.22.1 compatibility issues
        obj = at

        dyn = FIRE(obj)
        e_trace, f_trace = [], []

        def cb():
            e_trace.append(float(at.get_potential_energy()))
            f = at.get_forces()
            fmax_val = float(np.max(np.linalg.norm(f, axis=1)))
            f_trace.append(fmax_val)

        dyn.attach(cb, interval=1)
        if self.config.verbose:
            print(f"[ALIGNN] Relax: fmax={fmax}, steps={max_steps}")
        dyn.run(fmax=fmax, steps=max_steps)

        if self.config.verbose:
            e_final = float(at.get_potential_energy())
            f_final = float(np.max(np.linalg.norm(at.get_forces(), axis=1)))
            print(f"[ALIGNN] Done: E={e_final:+.4f} Fmax={f_final:.4f}")
        return at, e_trace, f_trace
