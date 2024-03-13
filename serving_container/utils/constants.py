from pathlib import Path

path = Path(__file__).resolve().parent.parent

SUBSET_DIR = path / "subset_data"
SUBSET_PATH = SUBSET_DIR / "test_data_subset.csv"

MODEL_DETAILS_BUCKET = "nashtech_vertex_ai_artifact"
MODEl_DETAILS_FILE_NAME = "validated_model.json"
fit_db_model_name = "db_scan"
SAVED_MODEL_BUCKET = "nashtech_vertex_ai_artifact"
