FROM python:3.10
WORKDIR /usr/src/app

COPY /src ./src
RUN pip install flask
RUN pip install --no-cache-dir -r src/requirements.txt

ENV AIP_HEALTH_ROUTE=/ping
ENV AIP_PREDICT_ROUTE=/predict

USER root

#ENTRYPOINT ["python", "src/serve_model.py"]
ENTRYPOINT ["sudo", "python", "src/serve_model.py"]
ENV PYTHONPATH /usr/src/app

