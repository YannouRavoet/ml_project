#NOTE: the open_spiel repository now has its own Dockerfile. You can build from there.
#Why use this Dockerfile?
# - runs on python3 => PyCharm can detect Interpreter Structure
# - smaller size
#source: https://github.com/theduelinghouse/open_spiel-docker
FROM python:3.6-slim-buster

#install sudo and git
RUN apt-get update && \
    apt-get upgrade -y && \
      apt-get -y install sudo git curl graphviz libgraphviz-dev
                                      #graphviz libgraphviz-dev: necessary to plot decision trees

# Copy requirements of what OpenSpiel depends on, provided by OpenSpiel
# This requires you to clone the deepmind/open_spiel repo into the ml_project folder
WORKDIR /
ADD . /ml_project
WORKDIR /ml_project/open_spiel

# Turn off apt-get interactions during installation
# 23/04/2020: With one of the latest updates of open_spiel the install.sh script redirects to ../open_spiel/scripts/install.sh
RUN sed -i -e 's/apt-get install/apt-get install -y/g' ./open_spiel/scripts/install.sh
RUN ./install.sh
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools testresources

#install OpenSpiel and run tests
RUN pip3 install .
RUN ./open_spiel/scripts/build_and_run_tests.sh

# Setup requirements for tournament.py
WORKDIR /ml_project
RUN pip3 install -r requirements.txt

WORKDIR /ml_project/open_spiel
ENTRYPOINT ["/bin/bash"]