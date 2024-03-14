from pathlib import Path

path = Path(__file__).resolve().parent.parent

SUBSET_DIR = path / "subset_data"
SUBSET_PATH = SUBSET_DIR / "test_data_subset.csv"

RESOURCE_BUCKET = "nashtech_vertex_ai_artifact"
MODEL_ID = "db_scan"
