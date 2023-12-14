FROM python:3.8
WORKDIR /usr/src/app

COPY /src ./src
RUN pip install --no-cache-dir -r src/requirements.txt

ENV AIP_HEALTH_ROUTE=/ping
ENV AIP_PREDICT_ROUTE=/predict

ENTRYPOINT ["python", "src/serve_model.py"]

ENV PYTHONPATH /usr/src/app
