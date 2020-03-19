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
    return pyspiel.create_matrix_game("battle_of_the_sexes", "Battle of the Sexes",                     #        Ballet  Movies
                                      ["Ballet", "Movies"], ["Ballet", "Movies"],                       #Ballet  2,1     0,0
                                      [[2, 0], [0, 1]],                                                 #Movies  0,0     1,2
                                      [[1, 0], [0, 2]])

def _rock_paper_scissors_easy():                                                                        #                 ZERO-SUM
    return pyspiel.create_matrix_game("rock_paper_scissors", "Rock Paper Scissors",                     #           Rock    Paper   Scissors
                                      ["Rock", "Paper",  "Scissors"],["Rock", "Paper",  "Scissors"],    #Rock        0,0    -1,1      1,-1
                                      [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],                             #Paper       1,-1    0,0     -1,1
                                      [[0, 1, -1], [-1, 0, 1], [1, -1, 0]])                             #Scissor    -1,1     1,-1     0,0

def _staghunt_easy():                                                                                   #       COORDINATION
    return pyspiel.create_matrix_game("staghunt", "StagHunt",                                           #        Stag       Hare
                                      ["Stag", "Hare"], ["Stag", "Hare"],                               #Stag     1,1       0,2/3
                                      [[1, 0], [2/3, 2/3]],                                             #Hare   2/3,0     2/3,2/3
                                      [[1, 2/3], [0, 2/3]])

#open_spiel/python/egt/visualization_test.py
from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python.egt.dynamics import SinglePopulationDynamics
from open_spiel.python.egt import visualization
import matplotlib.pyplot as plt
from dynamics import replicator, boltzmann_qlearning, boltzmann_faqlearning, LenientMultiPopulationDynamics, MultiPopulationDynamics

plot_labels = {"matrix_rps": ["Rock", "Paper", "Scissors"],
               "matrix_mp": ["Player 1: prob of choosing Heads", "Player 2: prob of choosing Heads"],
               "matrix_pd":["Player 1: prob of choosing Cooperate", "Player 2: prob of choosing Cooperate"],
               "battle_of_the_sexes":["Player 1: prob of choosing Ballet", "Player 2: prob of choosing Ballet"],
                "matrix_sh":["Player 1: prob of choosing Stag", "Player 2: prob of choosing Stag"],
               "prisonners_dilemma":["Player 1: prob of choosing Defect", "Player 2: prob of choosing Defect"],
               "staghunt":["Player 1: prob of choosing Stag", "Player 2: prob of choosing Stag"]}

def _phaseplot(game):
    is_2x2 = game.num_cols() == 2
    payoff_tensor = game_payoffs_array(game)
    dyn = LenientMultiPopulationDynamics(payoff_tensor, boltzmann_faqlearning) if is_2x2 \
        else SinglePopulationDynamics(payoff_tensor, replicator)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="2x2") if is_2x2 else fig.add_subplot(111, projection="3x3")
    # ax.streamplot(dyn, density=0.5)
    ax.quiver(dyn)
    if is_2x2:
        ax.set_xlabel(plot_labels[game.get_type().short_name][0])
        ax.set_ylabel(plot_labels[game.get_type().short_name][1])
    else:
        ax.set_labels(plot_labels[game.get_type().short_name])
    plt.title(game.get_type().long_name.upper())
    plt.show()



#TODO: het properste is wss om de bijhorende dynamics te gebruiken per algoritme
def _trajectoryplot(game, population_histories):
    is_2x2 = game.num_cols() == 2
    if is_2x2:
        payoff_tensor = game_payoffs_array(game)
        #dyn = MultiPopulationDynamics(payoff_tensor, boltzmann_qlearning)                               # TODO: eps = replicator / boltz = boltzmann_qlearning / faq = boltzmann_faqlearning
        dyn = LenientMultiPopulationDynamics(payoff_tensor, boltzmann_faqlearning, k=5)         # TODO: voor de lfaq plots
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="2x2")
        ax.quiver(dyn)
        for pop_hist in population_histories:
            x = [hist[0][0] for hist in pop_hist]                   # take the prob of choosing the first action for player 1
            y = [hist[1][0] for hist in pop_hist]                   # take the prob of choosing the first action for player 2
            plt.plot(x,y)                                           # plot each population
        plt.title(game.get_type().long_name.upper())
        plt.xlabel(plot_labels[game.get_type().short_name][0])
        plt.ylabel(plot_labels[game.get_type().short_name][1])
        plt.show()
    return

def _dynamics_kplot(k_values, games):
    """
    :param k_values: array of k-values to plot for
    :param games: array of (2x2) matrix games to plot for
    :return: a clean plot of the lenient faq dynamics for each game for all k-values
    """
    n = len(games)
    ks = len(k_values)
    plt.figure(figsize=(n*32, ks))
    for g_, game in enumerate(games):
        payoff_tensor = game_payoffs_array(game)
        for k_, k in enumerate(k_values):
            dyn = MultiPopulationDynamics(payoff_tensor, [boltzmann_faqlearning] *2)
            ax = plt.subplot2grid((n, ks), (g_, k_), projection="2x2")
            ax.quiver(dyn)
            # plt.title(game.get_type().long_name.upper())
            plt.xlabel("x")
            plt.ylabel("y")
    plt.show()