FROM python:3.10.6

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
  graphviz


COPY requirements_x86.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Copy over and install source code from this package.
COPY src ./src
COPY model_dir ./model_dir
COPY dataset ./dataset
COPY train_data ./train_data
COPY utils ./utils
COPY constants.py ./constants.py
COPY README.md ./README.md