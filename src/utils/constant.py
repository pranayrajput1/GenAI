from pathlib import Path
import pandas as pd
path = Path(__file__).parents[1]


data = path / 'data'
# Define constants
BUCKET_NAME = "nashtech_vertex_ai_artifact"
PROJECT_ID='nashtech-ai-dev-389315'
REGION='us-central1'
DATA_PATH = "gs://{}/housepp.csv".format(BUCKET_NAME)
MODEL_PATH = "gs://{}/house-model".format(BUCKET_NAME)
model_path= "src/model"
TARGET_COLUMN = "SalePrice"
df = pd.read_csv(DATA_PATH)
# print(df.columns)


