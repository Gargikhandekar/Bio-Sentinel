from src.config import load_config, project_path
from src.data.loader import load_raw_brfss, build_processed_frame, save_processed


if __name__ == "__main__":
    cfg = load_config()
    raw = load_raw_brfss(project_path(*cfg["dataset"]["raw_path"].split("/")))
    processed = build_processed_frame(raw)
    path = save_processed(processed, project_path(*cfg["dataset"]["processed_path"].split("/")))
    print("wrote", path, "rows", len(processed))
