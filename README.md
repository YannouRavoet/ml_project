# ml_project
A project focussed on the application of Game Theory, Reinforcement Learning and Deep Learning in a multi-agent setting using the Google DeepMind OpenSpiel library. This project was made in the context of the 'Machine Learning: Project' (H0T25A) course at the KU Leuven.


#Docker
The dockerfile starts from a tensorflow build with GPU-support and adds the OpenSpiel library (and dependencies).

    docker image build -t open_spiel .
To test whether everything runs, start a new container and run the tic_tac_toe example

    open_spiel/build/examples/example --game=tic_tac_toe
    
#Arguments
