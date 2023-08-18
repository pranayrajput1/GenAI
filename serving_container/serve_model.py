import os
from flask import Flask, request, jsonify
import logging
from serving_container.utils.helpers import get_model_tokenizer, get_memory_usage

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

"""Defining trained model path"""
model_path = "./trained_model/"

logging.info("Task: Loading Saved Model and Tokenizer ")
model, tokenizer = get_model_tokenizer(
    pretrained_model_name_or_path=model_path,
    gradient_checkpointing=True
)
get_memory_usage()


@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
    """
    Function to check health status of endpoint.
    """
    return {"status": "healthy"}


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
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
