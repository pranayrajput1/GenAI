import joblib
import json
import os
from flask import Flask, request, jsonify
from serving_container.utils.constants import SUBSET_PATH, MODEL_DETAILS_BUCKET, MODEl_DETAILS_FILE_NAME, \
    SAVED_MODEL_BUCKET
from serving_container.utils.input_handler import handle_json, gcs_file_download
import logging

logging.basicConfig(level=logging.INFO)

logging.info(f"Task: Downloading {MODEl_DETAILS_FILE_NAME} from: {MODEL_DETAILS_BUCKET}")
gcs_file_download(MODEL_DETAILS_BUCKET, MODEl_DETAILS_FILE_NAME)

with open(MODEl_DETAILS_FILE_NAME, 'rb') as model_file_name:
    model_detail = json.load(model_file_name)

model_name = model_detail["validated_model"]

logging.info(f"Task: Downloading {model_name}.joblib from: {SAVED_MODEL_BUCKET}")
gcs_file_download(SAVED_MODEL_BUCKET, f'{model_name}.joblib')

"""Defining trained model path"""
trained_model_file = f"./{model_name}.joblib"

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
