from flask import Flask, jsonify, request
from insert_text_vector.chroma_db_impl import resume_vec_insert
from insert_text_vector.text_structuring import process_resumes_structuring
from utils.helpers import download_files_from_bucket
from utils.constants import resume_bucket_path, resume_path, persistence_directory, structured_text_dir

app = Flask(__name__)


#@app.route('/ping-ingestion', methods=['GET'])
def health_check():
    """
    Function to check health status of endpoint.
    """
    return {"status": "healthy"}, 200


#@app.route('/update-vectors', methods=['PUT'])
def update_vector_database(input_command):
    try:
        #if request.method == 'PUT':
	if input_command == "yes":
            '''Download files'''
            response, response_code = download_files_from_bucket(resume_bucket_path, resume_path)
            if not response or response_code != 200:
                return jsonify({"response": response}), response_code

            '''Process resumes structuring'''
            response, response_code = process_resumes_structuring(resume_directory=resume_path)
            if not response or response_code != 200:
                return jsonify({"response": response}), response_code

            '''Resume vector insertion'''
            response, response_code = resume_vec_insert(persistence_directory, structured_text_dir)
            if not response or response_code != 200:
                return jsonify({"response": response}), response_code

            return jsonify({"response": "Updated Vector Database Successfully Completed"}), response_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500


one, two = update_vector_database("yes")
print(one)
#if __name__ == '__main__':
#    app.run(host='0.0.0.0', port=4040)
