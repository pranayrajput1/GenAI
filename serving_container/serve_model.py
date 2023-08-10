import os
from flask import Flask, request
import logging
import time

from serving_container.utils.helpers import get_model_tokenizer, get_time

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

"""Defining trained model path"""
model_path = "./trained_model/"

"""Reading trained model saved as joblib file"""
start_time = time.time()

model, tokenizer = get_model_tokenizer(
    pretrained_model_name_or_path=model_path,
    gradient_checkpointing=True
)

end_time = time.time()
logging.info(get_time(start_time, end_time))


@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
    """
    Function to check health status of endpoint.
    """
    return {"status": "healthy"}


@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def predict_labels():
    """
    Function to take query as input and return answer as output.
    """
    return 0


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
