#source: https://github.com/theduelinghouse/open_spiel-docker
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
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools testresources

#install OpenSpiel and run tests
RUN pip3 install .
RUN pip3 install nox
RUN nox -s tests


# Set WORKDIR as the output of the build
WORKDIR /ml_project/open_spiel
ENTRYPOINT ["/bin/bash"]