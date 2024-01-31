from flask import Flask, jsonify, request
from model.model import get_response, get_model_tokenizer
from retriever.retriever import get_ranking_resumes
from utils.constants import model_id

app = Flask(__name__)

initial_loaded_model, initial_loaded_tokenizer = get_model_tokenizer(model_id)


@app.route('/ping', methods=['GET'])
def health_check():
    """
    Function to check health status of endpoint.
    """
    return {"status": "healthy"}, 200


@app.route('/predict', methods=['POST'])
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


@app.route('/ranking', methods=['POST'])
def rank_resume():
    try:
        if request.method == 'POST':
            data = request.get_json()
            user_input = data['input']
            state = data['model_state']

            ranking_input_prompt = get_ranking_resumes(job_title=user_input["job_tile"],
                                                       desired_skills=user_input["desired_skills"])

            final_response = get_response(user_query=ranking_input_prompt, model=initial_loaded_model,
                                          tokenizer=initial_loaded_tokenizer, reload_state=state)

            return jsonify({
                "predictions": final_response
            }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    server_host = '0.0.0.0'
    server_port = 5050
    app.run(host=server_host, port=server_port)
