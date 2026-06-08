"""
One-command CLI: give it a QE input file or a folder -> relax + QE + plot -> done.

Usage:
    python3 -m ferro_pipeline.cli my_input.in
    python3 -m ferro_pipeline.cli my_qe_folder/
    python3 -m ferro_pipeline.cli my_qe_folder/ --no-qe
"""

import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from .config import Config
from .pipeline import run_pipeline


def resolve_input(path: str) -> str:
    """
    Accept either a .in file or a folder.
    If folder, auto-detect the first .in file inside.
    """
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.in")) + \
                glob.glob(os.path.join(path, "*.scf")) + \
                glob.glob(os.path.join(path, "*.scf.in"))
        if not files:
            # try any file that might be a QE input
            files = [f for f in glob.glob(os.path.join(path, "*"))
                     if os.path.isfile(f) and not f.endswith((".UPF", ".upf"))]
        if files:
            found = files[0]
            print(f"[Input] Using folder '{path}', auto-detected input: {os.path.basename(found)}")
            return found
        raise FileNotFoundError(
            f"No QE input file (.in) found in folder '{path}'.\n"
            f"Please provide a .in file directly or put one in the folder."
        )
    raise FileNotFoundError(f"'{path}' is neither a file nor a directory.")


def main():
    parser = argparse.ArgumentParser(
        description="Ferroelectric transition pipeline. Give it a QE input file or a folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="QE input file (.in) or a folder containing one")
    parser.add_argument("-o", "--output", default=None, help="Output directory (default: ferro_results)")
    parser.add_argument("-a", "--amplitude", type=float, default=None, help="Distortion amplitude in A (default: 0.1)")
    parser.add_argument("-n", "--n-interp", type=int, default=None, help="Number of interpolation images (default: 9)")
    parser.add_argument("--fmax", type=float, default=None, help="ALIGNN force convergence in eV/A (default: 0.05)")
    parser.add_argument("--no-qe", action="store_true", help="Skip QE, ALIGNN-FF only")
    parser.add_argument("--checkpoint", default=None, help="Finetuned ALIGNN-FF checkpoint")

    args = parser.parse_args()

    # Resolve input (file or folder)
    input_file = resolve_input(args.input)

    config = Config()
    config._input_path = input_file
    if args.output: config.output_dir = args.output
    if args.amplitude is not None: config.distortion_amplitude = args.amplitude
    if args.n_interp is not None: config.n_interp_images = args.n_interp
    if args.fmax is not None: config.fmax_alignn = args.fmax
    if args.no_qe: config.use_qe = False
    if args.checkpoint: config.checkpoint_path = args.checkpoint

    run_pipeline(config)


if __name__ == "__main__":
    main()
