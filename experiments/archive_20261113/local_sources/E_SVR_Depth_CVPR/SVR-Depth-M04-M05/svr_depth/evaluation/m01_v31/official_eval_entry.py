"""Launch a specified upstream evaluator only when its data root is unambiguous."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def checked_environment(data_root, environ):
    root = Path(data_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--dataroot must be a directory")
    old = environ.get("DATAROOT")
    if old and Path(old).expanduser().resolve() != root:
        raise ValueError("DATAROOT conflicts with --dataroot; fix the environment")
    env = dict(environ)
    env["DATAROOT"] = str(root)
    return root, env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--dataroot", required=True)
    args, rest = parser.parse_known_args(argv)
    root, env = checked_environment(args.dataroot, os.environ)
    repo = Path(args.official_repo).expanduser().resolve(strict=True)
    script = repo / "scripts" / "eval.py"
    if not script.is_file() or not (repo / "velodepth" / "datasets" / "base_dataset.py").is_file():
        raise ValueError("--official-repo must contain the upstream scripts/eval.py and datasets")
    # Ensure the supplied upstream checkout wins over an installed SVR-derived
    # velodepth package, without modifying the upstream repository.
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"Using upstream evaluator {script}; DATAROOT={root}", flush=True)
    return subprocess.run([sys.executable, str(script), "--dataroot", str(root), *rest],
                          cwd=repo, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
