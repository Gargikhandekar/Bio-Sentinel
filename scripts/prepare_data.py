"""Build the processed multi-label parquet from the raw BRFSS 2021 XPT file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, project_path  # noqa: E402
from src.data.loader import build_processed_frame, load_raw_brfss, save_processed  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    raw = load_raw_brfss(project_path(*cfg["dataset"]["raw_path"].split("/")))
    processed = build_processed_frame(raw)
    path = save_processed(processed, project_path(*cfg["dataset"]["processed_path"].split("/")))
    print("wrote", path, "rows", len(processed))
