#source: https://github.com/theduelinghouse/open_spiel-docker
#Start from tensorflow with gpu support
FROM tensorflow/tensorflow:latest-gpu-py3

#install sudo and git
RUN apt-get update && \
    apt-get upgrade -y && \
      apt-get -y install sudo git

# Copy requirements of what OpenSpiel depends on, provided by OpenSpiel
WORKDIR /
ADD . /ml_project
WORKDIR /ml_project
RUN git clone https://github.com/deepmind/open_spiel.git
WORKDIR /ml_project/open_spiel

# Turn off apt-get interactions during installation
RUN sed -i -e 's/apt-get install/apt-get install -y/g' ./install.sh
RUN ./install.sh
RUN pip3 install -r requirements.txt

# Install CMake with verion higher than 3.12
RUN pip3 install cmake

# Copy the directory with OpenSpiel source code
RUN ./open_spiel/scripts/build_and_run_tests.sh

# Set WORKDIR as the output of the build
WORKDIR /ml_project