# ml_project
A project focussed on the application of Game Theory, Reinforcement Learning and Deep Learning in a multi-agent setting using the Google DeepMind OpenSpiel library. This project was made in the context of the 'Machine Learning: Project' (H0T25A) course at the KU Leuven.


#Docker
The dockerfile starts from a tensorflow build with GPU-support and adds the OpenSpiel library (and dependencies).
  docker build -t open_spiel .
