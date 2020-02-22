#source: https://github.com/theduelinghouse/open_spiel-docker
#Option 1: NVIDIA GPU DEVICE with nvidia docker installed
#Start from tensorflow with gpu support
#FROM tensorflow/tensorflow:latest-gpu-py3
#Option 2: NO NVIDIA GPU DEVICE
#Start from tensorflow without gpu support
#FROM tensorflow/tensorflow:latest-py3
FROM python:3.6-slim-buster

#install sudo and git
RUN apt-get update && \
    apt-get upgrade -y && \
      apt-get -y install sudo git curl

# Copy requirements of what OpenSpiel depends on, provided by OpenSpiel
WORKDIR /
ADD . /ml_project
WORKDIR /ml_project
RUN git clone https://github.com/deepmind/open_spiel.git
WORKDIR /ml_project/open_spiel

# Turn off apt-get interactions during installation
RUN sed -i -e 's/apt-get install/apt-get install -y/g' ./install.sh
RUN ./install.sh
RUN virtualenv -p python3 venv
RUN /bin/bash -c "source venv/bin/activate"

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# Install pip deps as your user. Do not use the system's pip.
RUN python3 get-pip.py --user
RUN pip3 install --upgrade pip --user
RUN pip3 install --upgrade setuptools testresources --user

RUN python3 -m pip install .
RUN pip install nox
RUN nox -s tests
# Set WORKDIR as the output of the build
WORKDIR /ml_project