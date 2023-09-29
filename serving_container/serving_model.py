import joblib
import os
from flask import Flask, request, jsonify
from serving_container.utils.constants import SUBSET_PATH
from serving_container.utils.input_handler import handle_json, download_model_files_from_bucket
from google.cloud import storage
import logging

logging.basicConfig(level=logging.INFO)

"""Saved model gcs bucket"""
bucket_name = "dbscan-model"

logging.info("Task: Making Directory: trained_model if not exist for saving trained model from gcs bucket")
model_path = "downloaded_model"
os.makedirs(model_path, exist_ok=True)

file_name = "db_scan.joblib"

logging.info(f"Task: Downloading model files from GCS Bucket: {bucket_name}")
download_model_files_from_bucket(bucket_name, model_path)
logging.info("Task: Model Files Downloaded Successfully")

"""Defining trained model path"""
trained_model_file = f"{model_path}/{file_name}"

storage_client = storage.Client()
bucket = storage_client.get_bucket(bucket_name)

"""Reading trained model saved as joblib file"""
with open(trained_model_file, 'rb') as file:
    model = joblib.load(file)

app = Flask(__name__)


@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
    """
    Function to check health status of endpoint.
    """
    return {"status": "healthy"}, 200


@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def predict_labels():
    """
    Function to take instance as input dictionary return result as output.
    """
    response = []
    try:
        result = request.get_json(silent=True, force=True)
        data = result['instances']
        if len(data) == 0:
            return jsonify({"error": "No instances provided"}), 400
        else:
            for item in data:
                input_dataframe = handle_json([item], SUBSET_PATH)
                prediction = model.fit_predict(input_dataframe)
                prediction = prediction.tolist()
                output = prediction[-1]
                result = "Outlier" if output < 0 else "Not Outlier"
                response.append(result)

        return jsonify({
            "predictions": response
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
