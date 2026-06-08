"""
QE GPU engine.
Uses parameters from the original QE input file (pseudo_dir, ecutwfc, kpts...).
"""

import os, re, time, shutil, subprocess
from typing import Dict, Optional

from .config import Config
from .io_utils import write_qe_input


class QEGPUEngine:
    def __init__(self, config: Config, qe_params: dict):
        self.config = config
        self.qe_params = qe_params  # parsed from original input file
        self.pw_path = self._find_pwx()

    def _find_pwx(self) -> Optional[str]:
        # Look in project directory first (self-contained)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(project_root, "qe_src", "qe_src", "build", "bin", "pw.x"),
            os.path.join(project_root, "qe_src", "build", "bin", "pw.x"),
            os.path.join(project_root, "qe_src", "bin", "pw.x"),
            shutil.which("pw.x"),
        ]
        for c in candidates:
            if c and os.path.isfile(c):
                return c
        print("[QE] pw.x not found. QE refinement will be skipped.")
        return None

    @staticmethod
    def parse_energy(text: str) -> Optional[float]:
        for line in text.split("\n"):
            if "!" in line and "total energy" in line:
                parts = line.replace("=", " ").split()
                for p in parts:
                    try:
                        return float(p)
                    except ValueError:
                        continue
        return None

    def run_scf(self, atoms, job_name="scf_qe") -> Dict:
        if not self.pw_path:
            return {"energy": None, "time_s": 0, "ok": False}

        inp = write_qe_input(atoms, self.qe_params, job_name=job_name,
                             calculation="scf")

        inp_dir = os.path.join(self.config.output_dir, "qe_inputs")
        os.makedirs(inp_dir, exist_ok=True)
        inp_path = os.path.join(inp_dir, f"{job_name}.in")
        with open(inp_path, "w") as f:
            f.write(inp)

        cmd = [self.pw_path, "-i", os.path.abspath(inp_path),
               "-npool", str(self.config.qe_npool)]
        if self.config.verbose:
            print(f"[QE] Running: {' '.join(cmd)}")

        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")

        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=self.config.output_dir, env=env)
        t1 = time.time()

        # Save stdout to file for debugging
        out_path = inp_path.replace(".in", ".out")
        with open(out_path, "w") as f:
            f.write(r.stdout)
            f.write("\n\n--- STDERR ---\n")
            f.write(r.stderr)

        # Parse energy from stdout, then try output file if not found
        energy = self.parse_energy(r.stdout)
        if energy is None:
            energy = self.parse_energy(r.stderr)

        if self.config.verbose:
            if energy:
                e_ev = energy * 13.605698
                print(f"[QE] Energy = {energy:.8f} Ry ({e_ev:.6f} eV)")
            else:
                print(f"[QE] WARNING: energy not found in output. Check {out_path}")
            print(f"[QE] Time = {t1-t0:.1f}s")
        return {"energy": energy, "time_s": round(t1-t0, 2),
                "ok": r.returncode == 0, "stdout": r.stdout}
