import os
import sys
import numpy as np
import policy_handler
from open_spiel.python.algorithms.exploitability import exploitability
from open_spiel.python.policy import PolicyFromCallable

import matplotlib.pyplot as plt


# openspiel.python.examples.tic_tac_toe_qlearner.py
def command_line_action(time_step):
    """Gets a valid action from the user on the command line."""
    current_player = time_step.observations["current_player"]
    info_state = time_step.observations["info_state"][current_player]
    legal_actions = time_step.observations["legal_actions"][current_player]
    action = -1
    while action not in legal_actions:
        print("You are player {}".format(current_player))
        print("Infostate: [Pl0, Pl1, J  , Q  , K  , P0p, P0b, P1p, P1b, ..., ...]")
        print(f"Infostate: {info_state}")
        print("Choose an action from {}:".format(legal_actions))
        sys.stdout.flush()
        action_str = input()
        try:
            action = int(action_str)
        except ValueError:
            continue
    return action


# https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings-in-python
def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])



def plot_policies(game, prefix):
    """
    :param game: pyspiel Game class
    :param prefix: the prefix within the policies directory of the game directory (i.e. CFR/temp/temp_)
    :return: void
    """
    # get all the policies
    files = np.array([])

    for (root, subFolder, filenames) in os.walk('policies'):
        for file in filenames:
            path = os.path.join(root, file)
            if prefix in path:
                files = np.append(files, path)

    policies = {}
    for file in files:
        iterations = (int(file.split('_')[1]))
        policy = policy_handler.load_to_tabular_policy(file)
        policies[iterations] = PolicyFromCallable(game, policy)

    #get all the desired metrics
    exploitabilities = {}
    for key in policies:
        exploitabilities[key] = exploitability(game, policies[key])

    #sort the metrics
    exploitabilities = sorted(exploitabilities.items())
    iterations, exploitabilities = zip(*exploitabilities)

    #plot the metrics
    plt.plot(iterations, exploitabilities)
    plt.legend(['exploitability'])
    plt.title('{} - Metrics ifo training iterations'.format(str(game)))
    plt.xlabel('number of training iterations')
    plt.ylabel('metric values')
    plt.yscale('log')
    plt.show()

    return