FROM python:3.8

WORKDIR /root

COPY src /root/src
COPY requirements.txt /root/requirements.txt
COPY main.py /root/main.py

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]
