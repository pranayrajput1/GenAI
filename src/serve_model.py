from flask import Flask, jsonify, request
from model.model import get_response, get_model_tokenizer
from src.utils.helpers import download_files_from_bucket
from utils.constants import model_id, resume_bucket_path, download_resume_path

app = Flask(__name__)


# initial_loaded_model, initial_loaded_tokenizer = get_model_tokenizer(model_id)


@app.route('/ping', methods=['GET'])
# @app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
    """
    Function to check health status of endpoint.
    """
    return {"status": "healthy"}, 200


# @app.route('/predict', methods=['POST'])
# # @app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
# def serve_model():
#     try:
#         if request.method == 'POST':
#             data = request.get_json()
#             user_input = data['input']
#             state = data['model_state']
#             response = get_response(user_query=user_input, model=initial_loaded_model,
#                                     tokenizer=initial_loaded_tokenizer, reload_state=state)
#
#             return jsonify({
#                 "predictions": response
#             }), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


@app.route('/update-vectors', methods=['PUT'])
def update_vector_database():
    try:
        if request.method == 'PUT':
            response, response_code = download_files_from_bucket(resume_bucket_path, download_resume_path)

            if response and response_code == 200:
                return jsonify({"response": "Updated Vector Database Successfully Completed"}), response_code
            else:
                return jsonify({"response": response}), response_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
