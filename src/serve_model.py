from flask import Flask, jsonify, request
from model.model import get_response, get_model_tokenizer
from src.insert_text_vector.chroma_db_impl import resume_vec_insert
from src.insert_text_vector.text_structuring import process_resumes_structuring
from utils.helpers import download_files_from_bucket
from utils.constants import model_id, resume_bucket_path, resume_path, persistence_directory, structured_text_dir

app = Flask(__name__)

initial_loaded_model, initial_loaded_tokenizer = get_model_tokenizer(model_id)


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
            state = data['model_state']
            response = get_response(user_query=user_input, model=initial_loaded_model,
                                    tokenizer=initial_loaded_tokenizer, reload_state=state)

            return jsonify({
                "predictions": response
            }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/update-vectors', methods=['PUT'])
def update_vector_database():
    try:
        if request.method == 'PUT':
            '''Download files'''
            response, response_code = download_files_from_bucket(resume_bucket_path, resume_path)
            if not response or response_code != 200:
                return jsonify({"response": response}), response_code

            '''Process resumes structuring'''
            response, response_code = process_resumes_structuring(
                resume_directory=resume_path,
                input_model=initial_loaded_model,
                input_tokenizer=initial_loaded_tokenizer,
                model_reload_state=None)
            if not response or response_code != 200:
                return jsonify({"response": response}), response_code

            '''Resume vector insertion'''
            response, response = resume_vec_insert(persistence_directory, structured_text_dir)
            if not response or response != 200:
                return jsonify({"response": response}), response_code

            return jsonify({"response": "Updated Vector Database Successfully Completed"}), response_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
