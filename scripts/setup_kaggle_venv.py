from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"\n[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)


def resolve_venv_paths(venv_path: Path) -> tuple[Path, Path]:
    if os.name == "nt":
        bin_dir = venv_path / "Scripts"
    else:
        bin_dir = venv_path / "bin"
    return bin_dir / "python", bin_dir / "pip"


def clean_python_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def has_pip(venv_python: Path, env: dict[str, str]) -> bool:
    result = subprocess.run(
        [str(venv_python), "-S", "-m", "pip", "--version"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def bootstrap_pip_with_get_pip(venv_python: Path, env: dict[str, str]) -> None:
    url = "https://bootstrap.pypa.io/get-pip.py"
    with tempfile.TemporaryDirectory() as td:
        script_path = Path(td) / "get-pip.py"
        print(f"[INFO] Downloading {url} -> {script_path}")
        urllib.request.urlretrieve(url, script_path)
        run_cmd([str(venv_python), "-S", str(script_path)], env=env)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_req = repo_root / "requiment.txt"
    parser = argparse.ArgumentParser(description="Create isolated Kaggle venv and install project dependencies")
    parser.add_argument("--venv-path", type=Path, default=Path("/kaggle/working/venvs/pcb_env"))
    parser.add_argument("--requirements", type=Path, default=default_req)
    parser.add_argument("--skip-mmcv", action="store_true", help="Skip mmcv install via openmim")
    parser.add_argument("--kernel-name", type=str, default="pcb-venv", help="Jupyter kernel name")
    parser.add_argument(
        "--register-kernel",
        action="store_true",
        help="Register ipykernel so Kaggle notebook can select this venv kernel",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    req_path = args.requirements.resolve()
    venv_path = args.venv_path.resolve()

    if not req_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {req_path}")

    print(f"[INFO] Requirements: {req_path}")
    print(f"[INFO] Venv path:    {venv_path}")

    env = clean_python_env()
    run_cmd([sys.executable, "-S", "-m", "venv", str(venv_path), "--without-pip"], env=env)
    venv_python, _ = resolve_venv_paths(venv_path)

    if not has_pip(venv_python, env):
        bootstrap_pip_with_get_pip(venv_python, env)

    # Install wrapt first to avoid Kaggle sitecustomize import errors.
    run_cmd([str(venv_python), "-S", "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "wrapt"], env=env)
    run_cmd([str(venv_python), "-S", "-m", "pip", "install", "-r", str(req_path)], env=env)

    if not args.skip_mmcv:
        run_cmd([str(venv_python), "-S", "-m", "mim", "install", "mmcv>=2.1.0,<2.2.0"], env=env)

    if args.register_kernel:
        run_cmd([str(venv_python), "-S", "-m", "pip", "install", "ipykernel"], env=env)
        run_cmd(
            [
                str(venv_python),
                "-S",
                "-m",
                "ipykernel",
                "install",
                "--user",
                "--name",
                args.kernel_name,
                "--display-name",
                f"Python ({args.kernel_name})",
            ],
            env=env,
        )

    print("\n[INFO] Setup completed.")
    print(f"[INFO] Use this Python: {venv_python}")
    if os.name != "nt":
        print(f"[INFO] Activate: source {venv_path}/bin/activate")
    else:
        print(f"[INFO] Activate: {venv_path}\\Scripts\\Activate.ps1")


if __name__ == "__main__":
    main()
