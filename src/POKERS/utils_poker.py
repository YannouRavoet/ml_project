import os
import sys
import numpy as np
import policy_handler
from open_spiel.python.algorithms.exploitability import exploitability, nash_conv
from open_spiel.python.policy import PolicyFromCallable
from open_spiel.python.algorithms import cfr, fictitious_play

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



def plot_policies(game, algorithms):
    """
    :param game: pyspiel Game class
    :param algorithms: {string: string} maps the algorithm name to the prefix within the policies directory of the game directory (i.e. {'CFR': 'CFR/temp/temp_'})
    :return: void
    """
    exploitabilities = {}
    nash_convs = {}
    for algo in algorithms:
        algo_prefix = algorithms[algo]

        # get all the files
        files = np.array([])
        for (root, subFolder, filenames) in os.walk('policies'):
            for file in filenames:
                path = os.path.join(root, file)
                if algo_prefix in path:
                    files = np.append(files, path)
        # get the policy from each file
        algo_policies = {}
        for file in files:
            algo_iterations = (int(file.split(algo_prefix)[1]))
            algo_policy = policy_handler.load_to_tabular_policy(file)
            algo_policies[algo_iterations] = PolicyFromCallable(game, algo_policy)

        #get all the desired metrics of each policy
        algo_exploitabilities = {}
        algo_nashconvs = {}
        for key in algo_policies:
            algo_exploitabilities[key] = exploitability(game, algo_policies[key])
            algo_nashconvs[key] = nash_conv(game, algo_policies[key])
        exploitabilities[algo] = algo_exploitabilities
        nash_convs[algo] = algo_nashconvs

    # PLOTTING
    def plot_series(title, metric, series):
        legend = []
        for algo in series:
            algo_series = series[algo]
            algo_series = sorted(algo_series.items())
            algo_iterations, algo_series = zip(*algo_series)
            plt.plot(algo_iterations, algo_series)
            legend.append(algo)

        plt.title('{} - {}'.format(str(game), title))
        plt.legend(legend)
        plt.xlabel('number of training iterations')
        plt.ylabel(metric)
        plt.yscale('log')
        plt.show()

    plot_series('Exploitability ifo training iterations', 'Exploitability', exploitabilities)
    plot_series('NashConv ifo training iterations', 'NashConv', nash_convs)
    return

"""TRAINING ALGORITHMS"""

def CFR_Solving(game, iterations, save_every = 0, save_prefix = 'temp'):
    def save_cfr():
        policy = cfr_solver.average_policy()
        policy = dict(zip(policy.state_lookup, policy.action_probability_array))
        return policy_handler.save_to_tabular_policy(game, policy, "policies/CFR/{}/{}".format(save_prefix, it))

    cfr_solver = cfr.CFRSolver(game)
    for it in range(iterations+1):  #so that if you tell it to train 20K iterations, the last save isn't 19999
        cfr_solver.evaluate_and_update_policy()
        if save_every != 0 and it%save_every == 0: #order is important
            save_cfr()
    return save_cfr()

def CFRPlus_Solving(game, iterations, save_every = 0, save_prefix = 'temp'):
    def save_cfrplus():
        policy = cfr_solver.average_policy()
        policy = dict(zip(policy.state_lookup, policy.action_probability_array))
        return policy_handler.save_to_tabular_policy(game, policy, "policies/CFRPlus/{}/{}".format(save_prefix, it))

    cfr_solver = cfr.CFRPlusSolver(game)
    for it in range(iterations+1):  #so that if you tell it to train 20K iterations, the last save isn't 19999
        cfr_solver.evaluate_and_update_policy()
        if save_every != 0 and it%save_every == 0: #order is important
            save_cfrplus()
    return save_cfrplus()


def XFP_Solving(game, iterations, save_every = 0, save_prefix = 'temp'):
    def save_xfp():
        policy = xfp_solver.average_policy_tables()
        policy_keys = np.concatenate((list(policy[0].keys()), list(policy[1].keys())), 0)
        policy_values = np.concatenate((list(map(lambda d: list(d.values()), list(policy[0].values()))),
                                        list(map(lambda d: list(d.values()), list(policy[1].values())))), 0)
        #change possible None's into 0
        policy_values = [(d if d else 0 for d in a) for a in policy_values]
        policy = dict(zip(policy_keys, policy_values))
        return policy_handler.save_to_tabular_policy(game, policy, "policies/XFP/{}/{}".format(save_prefix, it))
    xfp_solver = fictitious_play.XFPSolver(game)
    for it in range(iterations+1):
        xfp_solver.iteration()
        if save_every != 0 and it%save_every == 0: #order is important
            save_xfp()
    return save_xfp()


def print_algorithm_results(game, policy, algorithm_name):
    print(algorithm_name.upper())
    callable_policy = PolicyFromCallable(game, policy)
    policy_exploitability = exploitability(game, callable_policy)
    # print(callable_policy._callable_policy.action_probability_array)
    print("exploitability = {}".format(policy_exploitability))
