import pyspiel
from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python.egt.dynamics import SinglePopulationDynamics
from open_spiel.python.egt import visualization
import matplotlib.pyplot as plt
from dynamics import replicator, boltzmann_qlearning, boltzmann_faqlearning, LenientMultiPopulationDynamics, MultiPopulationDynamics
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
                                      [[3, 3], [0, 5]],                                             #Silent  -12,0   -3,-3
                                      [[5, 0], [1,1]])

def _battle_of_the_sexes_easy():                                                                        #       COORDINATION
    return pyspiel.create_matrix_game("battle_of_the_sexes", "Battle of the Sexes",                     #        Ballet  Movies
                                      ["Ballet", "Movies"], ["Ballet", "Movies"],                       #Ballet  2,1     0,0
                                      [[2, 0], [0, 1]],                                                 #Movies  0,0     1,2
                                      [[1, 0], [0, 2]])

def _biased_rock_paper_scissors_easy():                                                                        #                 ZERO-SUM
    return pyspiel.create_matrix_game("biased_rock_paper_scissors", "Biased Rock Paper Scissors",                     #           Rock    Paper   Scissors
                                      ["Rock", "Paper",  "Scissors"],["Rock", "Paper",  "Scissors"],    #Rock        0,0    -3,3      1,-1
                                      [[0, -3, 1], [3, 0, -2], [-1, 2, 0]],                             #Paper       3,-3    0,0     -2,2
                                      [[0, 3, -1], [-3, 0, 2], [1, -2, 0]])                             #Scissor    -1,1     2,-2     0,0

def _staghunt_easy():                                                                                   #       COORDINATION
    return pyspiel.create_matrix_game("staghunt", "StagHunt",                                           #        Stag       Hare
                                      ["Stag", "Hare"], ["Stag", "Hare"],                               #Stag     1,1       0,2/3
                                      [[1, 0], [2/3, 2/3]],                                             #Hare   2/3,0     2/3,2/3
                                      [[1, 2/3], [0, 2/3]])


#open_spiel/python/egt/visualization_test.py
plot_labels = {"matrix_rps": ["Rock", "Paper", "Scissors"],
               "matrix_mp": ["Player 1: prob of choosing Heads", "Player 2: prob of choosing Heads"],
               "matrix_pd":["Player 1: prob of choosing Cooperate", "Player 2: prob of choosing Cooperate"],
               "battle_of_the_sexes":["Player 1: prob of choosing Ballet", "Player 2: prob of choosing Ballet"],
                "matrix_sh":["Player 1: prob of choosing Stag", "Player 2: prob of choosing Stag"],
               "prisonners_dilemma":["Player 1: prob of choosing Defect", "Player 2: prob of choosing Defect"],
               "staghunt":["Player 1: prob of choosing Stag", "Player 2: prob of choosing Stag"],
               "biased_rock_paper_scissors": ["Rock", "Paper", "Scissors"],}

def _phaseplot(games, bstreamplot = False):
    plt.figure(figsize=(32,len(games)))
    for g, game in enumerate(games):
        # SETUP VALUES
        is_2x2 = game.num_cols() == 2
        payoff_tensor = game_payoffs_array(game)
        #dynamics: choose between replicator, boltzmann_qlearning, boltzmann_faqlearning
        #for lfaq: LenientMultiPopulationDynamics(payoff_tensor, boltzmann_faqlearning, k=...)    (only valid for 2x2 games)
        dyn = MultiPopulationDynamics(payoff_tensor, replicator) if is_2x2 else SinglePopulationDynamics(payoff_tensor, replicator)

        # PLOTTING
        ax = plt.subplot2grid((1, len(games)), (0, g), projection="2x2") if is_2x2 else plt.subplot2grid((1, len(games)), (0, g), projection="3x3")
        ax.streamplot(dyn, density=0.75, color='black', linewidth=1) if bstreamplot else ax.quiver(dyn)

        if is_2x2:
            ax.set_xlabel(plot_labels[game.get_type().short_name][0])
            ax.set_ylabel(plot_labels[game.get_type().short_name][1])
        else:
            ax.set_labels(plot_labels[game.get_type().short_name])
        plt.title(game.get_type().long_name.upper())
    plt.show()



def _trajectoryplot(game, population_histories, k = 1):
    is_2x2 = game.num_cols() == 2
    if is_2x2:
        payoff_tensor = game_payoffs_array(game)
        dyn = MultiPopulationDynamics(payoff_tensor, replicator)                      # TODO: eps = replicator / boltz = boltzmann_qlearning / faq = boltzmann_faqlearning
        #dyn = LenientMultiPopulationDynamics(payoff_tensor, boltzmann_faqlearning, k=k)         # TODO: voor de lfaq plots
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
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.01, 1.01)
        plt.show()
    return

def _dynamics_kplot(k_values, games):
    """
    :param k_values: array of k-values to plot for
    :param games: array of (2x2) matrix games to plot for
    :return: a clean plot of the lenient faq dynamics for each game for all k-values
    """
    games = [game for game in games if game.num_cols() == 2]
    n = len(games) +1                   #+1 to add k values
    ks = len(k_values) +1               #+1 to add game name plot
    plt.figure(figsize=(n*8, ks*4))
    for g_, game in enumerate(games):
        payoff_tensor = game_payoffs_array(game)
        #GAME TITLE
        ax = plt.subplot2grid((n, ks), (g_, 0))
        plt.text(1,0.5,game.get_type().long_name.upper(), fontsize=24, horizontalalignment='right', fontweight='bold')
        plt.axis('off')
        for k_, k in enumerate(k_values):
            dyn = MultiPopulationDynamics(payoff_tensor, boltzmann_faqlearning)
            ax = plt.subplot2grid((n, ks), (g_, k_+1), projection="2x2")
            ax.quiver(dyn)
            # plt.title(game.get_type().long_name.upper())
            plt.xlabel("x")
            plt.ylabel("y")
    #K-LABELS
    for k_, k in enumerate(k_values):
        ax = plt.subplot2grid((n, ks), (n-1, k_+1))
        plt.text(0.5, 1, "k = " + str(k), fontsize=24, horizontalalignment='center', verticalalignment='top', fontweight='bold')
        plt.axis('off')
    plt.show()