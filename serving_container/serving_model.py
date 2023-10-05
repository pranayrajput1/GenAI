import joblib
import json
import pandas as pd
from flask import Flask, request, jsonify
from serving_container.utils.constants import SUBSET_PATH, MODEL_DETAILS_BUCKET, MODEl_DETAILS_FILE_NAME, \
    SAVED_MODEL_BUCKET
from serving_container.utils.input_handler import handle_json, gcs_file_download, handle_file
import logging

logging.basicConfig(level=logging.INFO)

# logging.info(f"Task: Downloading {MODEl_DETAILS_FILE_NAME} from: {MODEL_DETAILS_BUCKET}")
# gcs_file_download(MODEL_DETAILS_BUCKET, MODEl_DETAILS_FILE_NAME)
#
# with open(MODEl_DETAILS_FILE_NAME, 'rb') as model_file_name:
#     model_detail = json.load(model_file_name)
#
# model_name = model_detail["validated_model"]
#
# logging.info(f"Task: Downloading {model_name}.joblib from: {SAVED_MODEL_BUCKET}")
# gcs_file_download(SAVED_MODEL_BUCKET, f'{model_name}.joblib')

model_name = "db_scan"
"""Defining trained model path"""
trained_model_file = f"./{model_name}.joblib"

"""Reading trained model saved as joblib file"""
with open(trained_model_file, 'rb') as file:
    model = joblib.load(file)

app = Flask(__name__)


# @app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
@app.route('/ping', methods=['GET'])
def health_check():
    """
    Function to check health status of endpoint.
    """
    return {"status": "healthy"}, 200


# @app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict_labels():
    """
    Function to predict labels based on input data (JSON or CSV) and return the results.
    """

    try:
        if request.is_json:
            result = request.get_json(silent=True, force=True)
            data = result['instances']

            if len(data) == 0:
                return jsonify({"error": "No instances provided"}), 400

            else:
                input_dataframe = handle_json(data, SUBSET_PATH)
                predictions = model.fit_predict(input_dataframe)

                input_dataframe['Prediction'] = ["Outlier" if output < 0 else "Not_Outlier" for output in predictions]
                final_df = input_dataframe[-len(data):]

                outlier_rows = final_df[final_df['Prediction'] == "Outlier"]
                non_outlier_rows = final_df[final_df['Prediction'] == "Not_Outlier"]

                outlier_list = outlier_rows.to_dict(orient='records')
                non_outlier_list = non_outlier_rows.to_dict(orient='records')

                response_data = {
                    "outliers": outlier_list,
                    "non_outliers": non_outlier_list
                }

                return jsonify({"predictions": response_data}), 200

        else:
            file = request.files.get('file')
            if not file:
                return jsonify({"error": "No CSV file provided"}), 400

            input_dataframe = handle_file(file)
            if input_dataframe.empty:
                return jsonify({"error": "CSV file is empty"}), 400

            predictions = model.fit_predict(input_dataframe)

            input_dataframe['Prediction'] = ["Outlier" if output < 0 else "Not Outlier" for output in predictions]

            outlier_rows = input_dataframe[input_dataframe['Prediction'] == "Outlier"]
            non_outlier_rows = input_dataframe[input_dataframe['Prediction'] == "Not Outlier"]

            outlier_list = outlier_rows.to_dict(orient='records')
            non_outlier_list = non_outlier_rows.to_dict(orient='records')

            response_data = {
                "outliers": outlier_list,
                "non_outliers": non_outlier_list
            }

            return jsonify({"predictions": response_data}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
