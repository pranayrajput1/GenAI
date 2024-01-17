from flask import Flask, jsonify, request
from model.model import get_response
import os

from src.decimal_model.decimal_model import get_decimal_pipeline, get_decimal_response
from src.utils.constants import decimal_model_name, system_prompt

app = Flask(__name__)

decimal_pipeline, tokenizer = get_decimal_pipeline(decimal_model_name)


@app.route('/ping', methods=['GET'])
# @app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
    """
    Function to check health status of endpoint.
    """
    return {"status": "healthy"}, 200


@app.route('/predict', methods=['POST'])
# @app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def serve_model():
    try:
        if request.method == 'POST':
            data = request.get_json()
            user_input = data['input']
            response = get_decimal_response(deci_generator=decimal_pipeline,
                                            tokenizer=tokenizer,
                                            system_prompt=system_prompt,
                                            user_prompt=user_input)

            return jsonify({
                "predictions": response
            }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
