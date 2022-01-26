FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install pkg-config
RUN rm -rf /var/cache/apt/archives

RUN pip install pip --upgrade
COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt
RUN rm ./requirements.txt