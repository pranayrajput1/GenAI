import os
from flask import Flask, request, jsonify
import logging
from serving_container.utils.helpers import get_model_tokenizer, get_memory_usage, download_model_files_from_bucket

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

"""Download model from gcs bucket"""
bucket_name = "llm_dolly_model"

logging.info("Task: Making Directory: trained_model if not exist for saving trained model from gcs bucket")
model_path = "trained_model"
os.makedirs(model_path, exist_ok=True)

logging.info(f"Task: Downloading model files from GCS Bucket: {bucket_name}")
download_model_files_from_bucket(bucket_name, model_path)
logging.info("Task: Model Files Downloaded Successfully")

logging.info(f"Task: Loading Saved Model and Tokenizer from local: {model_path} directory")
model, tokenizer = get_model_tokenizer(
    pretrained_model_name_or_path=model_path,
    gradient_checkpointing=True
)
get_memory_usage()
logging.info("Model Loaded Successfully")


@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
    """
    Function to check health status of endpoint.
    """
    return {"status": "healthy"}, 200


@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def predict_answer():
    """
    Function to take query as input and return answer as output.
    """
    try:
        query = request.get_json(silent=True, force=True)
        data = query['instances']

        if 'input' not in data:
            return jsonify({'error': 'Input data is missing'}), 400

        input_text = data['input']
        response = {'message': f'Received input: {input_text}'}
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
