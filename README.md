# Machine Learning: Project
A project focused on the application of Game Theory, Reinforcement Learning and Deep Learning in a Multi-Agent setting using the Google DeepMind OpenSpiel library. This project was made in the context of the 'Machine Learning: Project' (H0T25A) course at the KU Leuven. The open_spiel and ML_project20 directories are links to separate github repositories and should be cloned separately. The total download of the project then looks as follows:

       git clone https://github.com/YannouRavoet/ml_project
       cd ml_project
       git clone https://github.com/deepmind/open_spiel
       git clone https://github.com/probberechts/ML_project20

### Structure
The ML_Project20 directory contains the code for creating csv files. The open_spiel directory contains the open_spiel library. The resources directory contains all the plots as well as the two final reports (and some other stuff like plannings, etc...). The src directory is where the magic happens.

### Docker
The dockerfile starts from a python:3.6-slim-buster build and adds the OpenSpiel library (and dependencies).

    docker image build -t open_spiel .
To test whether everything runs, start a new container and run the tic_tac_toe example. 
Also test whether you can use the pyspiel library and modules inside your code.

    # testing a built example
    open_spiel/build/examples/example --game=tic_tac_toe
    
    # open python shell
    # import pyspiel and modules inside a python shell
    import pyspiel
    from open_spiel.python import rl_environment

### Pycharm - Docker + SciView
To be able to see plots in the SciView tab of PyCharm whilst using a docker container you need to:

    1. Go into the run configuration of the file that builds the plots
    2. Change the 'Docker container settings' tab as follows:
        a. Network mode: host
        b. Volume bindings += 
            Container path:     /tmp/.X11-unix 
            Host path:          /tmp/.X11-unix
        c. Environment variables +=
            Name:               DISPLAY
            Value:              :0
