"""Reproduce every reported number and figure with one command.

    python scripts/reproduce.py [--config path/to/config.yaml]

Needs data/raw/LLCP2021.XPT (see data/README.md). Seeds come from the config.
Writes results/metrics/summary.{json,csv,md} and results/figures/*.png.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, results_dir  # noqa: E402
from src.evaluation.experiments import run_all_experiments  # noqa: E402
from src.evaluation.report import make_figures, write_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    payload = run_all_experiments(args.config)
    out = results_dir(cfg)
    paths = write_summary(payload, out / "metrics")
    figures = make_figures(payload, out)
    print(paths["md"].read_text(encoding="utf-8"))
    print("wrote", *paths.values(), *figures, sep="\n  ")


if __name__ == "__main__":
    main()
