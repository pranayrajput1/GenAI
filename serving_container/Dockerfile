FROM python:3.10.6
WORKDIR /usr/src/app

COPY /serving_container ./serving_container
RUN pip install --no-cache-dir -r serving_container/requirements.txt


ENV AIP_MODEL_DIR=gs://fine_tuned_model/

#ENV AIP_HEALTH_ROUTE=/ping
#ENV AIP_PREDICT_ROUTE=/predict
#ENV AIP_HTTP_PORT=8080

#ENTRYPOINT ["python", "serving_container/serve_model.py"]

ENV PYTHONPATH /usr/src/app