import pyspiel
#open_spiel/python/examples/matrix_game_example.py
#ALL GAMES ARE SIMULTANEOUS, DETERMINISTIC, ONE-SHOT with a TERMINAL REWARD
def _matching_pennies_easy():                                                                           #           ZERO-SUM
    return pyspiel.create_matrix_game("matching_pennies", "Matching Pennies",                           #        Heads   Tails
                                      ["Heads", "Tails"], ["Heads", "Tails"],                           #Heads   -1,1    1,-1
                                      [[-1, 1], [1, -1]],                                               #Tails   1,-1    -1,1
                                      [[1, -1], [-1, 1]])

def _prisonners_dilemma_easy():                                                                         #       NON ZERO-SUM
    return pyspiel.create_matrix_game("prisonners_dilemma", "Prisoners Dilemma",                        #        Talk    Silent
                                      ["Talk", "Silent"], ["Talk", "Silent"],                           #Talk    -6,-6   0,-12
                                      [[-6, 0], [-12, -3]],                                             #Silent  -12,0   -3,-3
                                      [[-6, -12], [0, -3]])

def _battle_of_the_sexes_easy():                                                                        #       COORDINATION
    return pyspiel.create_matrix_game("battle_of_the_sexes", "Battle of the Sexes",                     #        Ballet  Fight
                                      ["Ballet", "Fight"], ["Ballet", "Fight"],                         #Ballet  2,1     0,0
                                      [[2, 0], [0, 1]],                                                 #Fight   0,0     1,2
                                      [[1, 0], [0, 2]])

def _rock_paper_scissors_easy():                                                                        #                 ZERO-SUM
    return pyspiel.create_matrix_game("rock_paper_scissors", "Rock Paper Scissors",                     #           Rock    Paper   Scissors
                                      ["Rock", "Paper",  "Scissors"],["Rock", "Paper",  "Scissors"],    #Rock        0,0    -1,1      1,-1
                                      [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],                             #Paper       1,-1    0,0     -1,1
                                      [[0, 1, -1], [-1, 0, 1], [1, -1, 0]])                             #Scissor    -1,1     1,-1     0,0


#open_spiel/python/egt/visualization_test.py
from open_spiel.python.egt import visualization, dynamics, utils
from matplotlib.pyplot import figure, title, xlabel, ylabel, show

#TODO: Add labels to RPS plot
plot_labels = {"matrix_rps": [],
               "matrix_mp": ["Player 1: prob of choosing Heads", "Player 2: prob of choosing Heads"],
               "matrix_pd":["Player 1: prob of choosing Cooperate", "Player 2: prob of choosing Cooperate"],
               "battle_of_the_sexes":["Player 1: prob of choosing Ballet", "Player 2: prob of choosing Ballet"],
                "matrix_sh":["Player 1: prob of choosing Stag", "Player 2: prob of choosing Stag"]}
def _phaseplot(game):
    is_2x2 = game.num_cols() == 2
    payoff_tensor = utils.game_payoffs_array(game)
    dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator) if is_2x2 \
        else dynamics.SinglePopulationDynamics(payoff_tensor, dynamics.replicator)
    fig = figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="2x2") if is_2x2 else fig.add_subplot(111, projection="3x3")
    #ax.streamplot(dyn)
    ax.quiver(dyn)
    title(game.get_type().long_name.upper())
    if is_2x2:
         xlabel(plot_labels[game.get_type().short_name][0])
         ylabel(plot_labels[game.get_type().short_name][1])
    show()


import numpy as np
def _trajectoryplot(game):
    return
