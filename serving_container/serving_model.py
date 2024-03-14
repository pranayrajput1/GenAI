import joblib
import os
from flask import Flask, request, jsonify
from serving_container.utils.constants import SUBSET_PATH, RESOURCE_BUCKET, MODEL_ID
from serving_container.utils.input_handler import handle_json, gcs_file_download
import logging

logging.basicConfig(level=logging.INFO)


logging.info(f"Task: Downloading {MODEL_ID}.joblib from: {RESOURCE_BUCKET}")
gcs_file_download(RESOURCE_BUCKET, f'{MODEL_ID}.joblib')

"""Defining trained model path"""
trained_model_file = f"./{MODEL_ID}.joblib"

"""Reading trained model saved as joblib file"""
with open(trained_model_file, 'rb') as file:
    model = joblib.load(file)

app = Flask(__name__)


@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
# @app.route('/ping', methods=['GET'])
def health_check():
    """
    Function to check health status of endpoint.
    """
    return {"status": "healthy"}, 200


@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
# @app.route('/predict', methods=['POST'])
def predict_labels():
    try:
        response_data = []
        result = request.get_json(silent=True, force=True)
        data = result['instances']

        if len(data) == 0:
            return jsonify({"error": "No instances provided"}), 400

        else:
            input_dataframe = handle_json(data, SUBSET_PATH)
            predictions = model.fit_predict(input_dataframe)

            input_dataframe['Prediction'] = ["Outlier" if output < 0 else "Not Outlier" for output in predictions]
            final_df = input_dataframe[-len(data):]

            for _, row in final_df.iterrows():
                request_data = {
                    "Global_intensity": row["Global_intensity"],
                    "Global_reactive_power": row["Global_reactive_power"]
                }

                prediction = row["Prediction"]

                result_dict = {
                    "request": request_data,
                    "response": prediction
                }

                response_data.append(result_dict)

        return jsonify({
            "predictions": response_data
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
