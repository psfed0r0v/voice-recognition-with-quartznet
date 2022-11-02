FROM python:3.9-slim-buster

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
COPY ./app /code/app
COPY ./config.yaml /code/config.yaml

RUN apt-get -y update && \
    apt-get -y install wget \
                       sudo \
                       software-properties-common \
                       tree \
                       bash-completion

RUN apt-get -y install libsndfile1 ffmpeg

RUN pip install -r /code/requirements.txt
RUN ls /code/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
