"""
Main pipeline: one function to run everything, output CSV data for you to plot.
"""

import os, json, time
import numpy as np
import pandas as pd
from datetime import datetime

from .config import Config
from .alignn_engine import AlignNNFEEngine
from .qe_engine import QEGPUEngine
from .io_utils import read_structure, write_poscar
from .pathway import add_ferroelectric_distortion, linear_interpolate


def run_pipeline(config: Config):
    """
    Ferroelectric transition pipeline: relax + QE -> CSV data.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    t_start = time.time()
    summary = {"status": "ok"}

    # ================================================================
    #  0. Read input structure + QE parameters from input file
    # ================================================================
    print("=" * 60)
    print("  Ferroelectric Transition Pipeline")
    print("  ALIGNN-FF (GPU) + QE (GPU)")
    print("=" * 60)

    input_path = getattr(config, "_input_path", None)
    if not input_path:
        raise ValueError("No input structure provided")

    print(f"\n[0] Reading structure: {input_path}")
    para = read_structure(input_path)
    n_atoms = len(para)
    cell = para.get_cell()
    print(f"    Atoms: {n_atoms}")
    print(f"    Cell:  {cell.lengths()}")

    # Parse QE parameters from input file (pseudo_dir, ecutwfc, kpts, etc.)
    from .io_utils import parse_qe_input
    qe_params = parse_qe_input(input_path)
    if qe_params:
        print(f"    QE params from input:")
        if "pseudo_dir" in qe_params:
            print(f"      pseudo_dir = {qe_params['pseudo_dir']}")
        if "ecutwfc" in qe_params:
            print(f"      ecutwfc    = {qe_params['ecutwfc']} Ry")
        if "ecutrho" in qe_params:
            print(f"      ecutrho    = {qe_params['ecutrho']} Ry")
        if "kpts" in qe_params:
            print(f"      kpts       = {qe_params['kpts']}")
        if "conv_thr" in qe_params:
            print(f"      conv_thr   = {qe_params['conv_thr']}")

    # ================================================================
    #  1. Build FE distortion
    # ================================================================
    print(f"\n[1] Building FE distortion (amplitude={config.distortion_amplitude} A)")
    atoms_i, atoms_f = add_ferroelectric_distortion(para, config.distortion_amplitude)

    # ================================================================
    #  2. ALIGNN-FF coarse relaxation
    # ================================================================
    print("\n[2] ALIGNN-FF coarse relaxation (GPU) ...")
    ff = AlignNNFEEngine(config)
    print("    Relaxing initial FE state ...")
    relaxed_i, ei_trace, fi_trace = ff.relax(atoms_i)
    print("    Relaxing final FE state ...")
    relaxed_f, ef_trace, ff_trace = ff.relax(atoms_f)

    write_poscar(relaxed_i, os.path.join(config.output_dir, "relaxed_initial.POSCAR"))
    write_poscar(relaxed_f, os.path.join(config.output_dir, "relaxed_final.POSCAR"))

    # ================================================================
    #  3. Interpolation + energy scan
    # ================================================================
    n_img = config.n_interp_images
    print(f"\n[3] Path interpolation ({n_img} images) + ALIGNN scan ...")

    images = linear_interpolate(relaxed_i, relaxed_f, n_img)
    rows = []
    for idx, im in enumerate(images):
        e, f = ff.get_energy_forces(im)
        fmax = float(np.max(np.linalg.norm(f, axis=1)))
        vol = im.get_volume()
        c = im.get_cell()
        rows.append({
            "image": idx,
            "t": idx / max(n_img - 1, 1),
            "energy_eV": e,
            "energy_per_atom_eV": e / n_atoms,
            "fmax_eV_per_Ang": fmax,
            "volume_Ang3": vol,
            "a_Ang": c[0, 0],
            "b_Ang": c[1, 1],
            "c_Ang": c[2, 2],
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(config.output_dir, "ferro_path_data.csv")
    df.to_csv(csv_path, index=False, float_format="%.8f")
    print(f"    Data saved: {csv_path}")

    e_arr = df["energy_eV"].values
    barrier = float(e_arr.max() - e_arr.min())
    i_max = int(e_arr.argmax())
    print(f"    Barrier (ALIGNN): {barrier:.4f} eV  (peak at image {i_max})")

    if config.save_intermediate:
        path_dir = os.path.join(config.output_dir, "path_structures")
        os.makedirs(path_dir, exist_ok=True)
        for idx, im in enumerate(images):
            write_poscar(im, os.path.join(path_dir, f"image_{idx:03d}.POSCAR"))

    # ================================================================
    #  4. QE GPU refinement (using original QE parameters)
    # ================================================================
    qe_df = None
    if config.use_qe:
        print("\n[4] QE GPU refinement (using params from input file) ...")
        qe = QEGPUEngine(config, qe_params)
        if qe.pw_path:
            keys = {0, n_img - 1, i_max}
            for d in (-2, -1, 1, 2):
                if 0 <= i_max + d < n_img:
                    keys.add(i_max + d)
            keys = sorted(keys)
            print(f"    Refining {len(keys)} key images: {keys}")

            qe_rows = []
            for idx in keys:
                im = images[idx]
                res = qe.run_scf(im, job_name=f"image_{idx:03d}")
                e_qe = res["energy"] * 13.605698 if res["energy"] else None
                t_val = idx / max(n_img - 1, 1)
                qe_rows.append({
                    "image": idx, "t": t_val,
                    "energy_Ry": res["energy"],
                    "energy_eV": e_qe,
                    "time_s": res["time_s"],
                })
                if e_qe:
                    print(f"      image {idx}: E = {e_qe:.6f} eV")
            qe_df = pd.DataFrame(qe_rows)
            qe_csv = os.path.join(config.output_dir, "qe_refined.csv")
            qe_df.to_csv(qe_csv, index=False, float_format="%.8f")
            print(f"    QE data saved: {qe_csv}")

    # ================================================================
    #  5. Summary
    # ================================================================
    t_total = time.time() - t_start
    summary.update({
        "timestamp": datetime.now().isoformat(),
        "total_time_s": round(t_total, 1),
        "n_atoms": n_atoms,
        "n_interp_images": n_img,
        "barrier_eV_alignn": round(barrier, 6),
        "barrier_peak_image": i_max,
        "input_structure": input_path,
        "qe_params_used": {
            "pseudo_dir": qe_params.get("pseudo_dir", ""),
            "ecutwfc": qe_params.get("ecutwfc", ""),
            "ecutrho": qe_params.get("ecutrho", ""),
            "kpts": qe_params.get("kpts", []),
        },
    })

    rpt_path = os.path.join(config.output_dir, "report.json")
    with open(rpt_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)
    print(f"  Barrier:  {barrier:.4f} eV")
    print(f"  Data:     {csv_path}")
    print(f"  Time:     {t_total:.1f}s")
    print("=" * 60)

    return summary
