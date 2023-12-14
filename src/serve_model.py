from flask import Flask, jsonify, request
from src.model.model import get_response
import os

app = Flask(__name__)


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
            response = get_response(user_input)

            return jsonify({
                "predictions": response
            }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
