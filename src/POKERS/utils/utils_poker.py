import os
import sys
import numpy as np
import policy_handler

from open_spiel.python.algorithms.exploitability import exploitability, nash_conv
from open_spiel.python.policy import PolicyFromCallable, tabular_policy_from_policy
from open_spiel.python.algorithms import cfr, fictitious_play, deep_cfr, cfr_br
import tensorflow as tf
import six

import tensorflow as tf

import matplotlib.pyplot as plt

from open_spiel.python.algorithms.exploitability import exploitability, nash_conv
from open_spiel.python import policy
from open_spiel.python.algorithms import cfr, fictitious_play, policy_gradient, nfsp, discounted_cfr
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import get_all_states

# openspiel.python.examples.tic_tac_toe_qlearner.py
def command_line_action(time_step):
    """Gets a valid action from the user on the command line."""
    current_player = time_step.observations["current_player"]
    info_state = time_step.observations["info_state"][current_player]
    legal_actions = time_step.observations["legal_actions"][current_player]
    action = -1
    while action not in legal_actions:
        print("You are player {}".format(current_player))
        print("Infostate: [Pl0, Pl1, J  , Q  , K  , P0p, P0b, P1p, P1b, P0pbp, P0pbb]")
        print(f"Infostate: {info_state}")
        print("Choose an action from {}:".format(legal_actions))
        sys.stdout.flush()
        action_str = input()
        try:
            action = int(action_str)
        except ValueError:
            continue
    return action

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
            algo_policies[algo_iterations] = policy.PolicyFromCallable(game, algo_policy)

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

def CFR_Solving(game, iterations, save_every = 0, save_prefix = 'temp', load_from_policy=None, load_from_policy_iterations = 0):
    class CFR_Solver_WithInit(cfr.CFRSolver):
        def __init__(self, game, current_policy):
            super(CFR_Solver_WithInit, self).__init__(game)
            self._current_policy = current_policy
            self._average_policy = self._current_policy.__copy__()
            self._initialize_info_state_nodes(self._root_node)

    def save_cfr():
        policy = cfr_solver.average_policy()
        policy = dict(zip(policy.state_lookup, policy.action_probability_array))
        policy_handler.save_to_tabular_policy(game, policy, "policies/CFR/{}/{}".format(save_prefix, it))

    cfr_solver = cfr.CFRSolver(game)
    # cfr_solver = CFR_Solver_WithInit(game, load_from_policy) if load_from_policy is not None else cfr.CFRSolver(game)

    for it in range(load_from_policy_iterations, load_from_policy_iterations+iterations+1):  #so that if you tell it to train 20K iterations, the last save isn't 19999
        if save_every != 0 and it%save_every == 0: #order is important
            save_cfr()
        cfr_solver.evaluate_and_update_policy()
    save_cfr()

def DCFR_Solving(game, iterations, a=3/2,b=0,g=2, save_every = 0, save_prefix = 'temp', load_from_policy=None, load_from_policy_iterations = 0):

    def save_dcfr():
        policy = dcfr_solver.average_policy()
        policy = dict(zip(policy.state_lookup, policy.action_probability_array))
        policy_handler.save_to_tabular_policy(game, policy, "policies/DCFR/{}/{}".format(save_prefix, it))

    dcfr_solver = discounted_cfr.DCFRSolver(game, alpha=a, beta=b,gamma=g)
    # cfr_solver = CFR_Solver_WithInit(game, load_from_policy) if load_from_policy is not None else cfr.CFRSolver(game)

    for it in range(load_from_policy_iterations, load_from_policy_iterations+iterations+1):  #so that if you tell it to train 20K iterations, the last save isn't 19999
        if save_every != 0 and it%save_every == 0: #order is important
            save_dcfr()
        dcfr_solver.evaluate_and_update_policy()
    save_dcfr()


def CFR_BR_Solving(game, iterations, save_every = 0, save_prefix = 'temp', load_from_policy=None, load_from_policy_iterations = 0):
    class CFR_Solver_WithInit(cfr_br.CFRBRSolver):
        def __init__(self, game, current_policy):
            super(CFR_Solver_WithInit, self).__init__(game)
            self._current_policy = current_policy
            self._average_policy = self._current_policy.__copy__()
            self._initialize_info_state_nodes(self._root_node)

    def save_cfr_br():
        policy = cfr_solver.average_policy()
        policy = dict(zip(policy.state_lookup, policy.action_probability_array))
        policy_handler.save_to_tabular_policy(game, policy, "policies/CFRBR/{}/{}".format(save_prefix, it))

    cfr_solver = cfr_br.CFRBRSolver(game)
    # cfr_solver = CFR_Solver_WithInit(game, load_from_policy) if load_from_policy is not None else cfr.CFRSolver(game)

    for it in range(load_from_policy_iterations, load_from_policy_iterations+iterations+1):  #so that if you tell it to train 20K iterations, the last save isn't 19999
        if save_every != 0 and it%save_every == 0: #order is important
            save_cfr_br()
        cfr_solver.evaluate_and_update_policy()
    save_cfr_br()


def CFRPlus_Solving(game, iterations, save_every = 0, save_prefix = 'temp', load_avg_policy = None, load_cur_policy = None, load__iterations=0):
    class CFRPlus_Solver_WithInit(cfr.CFRPlusSolver):
        def __init__(self, game):
            super(CFRPlus_Solver_WithInit, self).__init__(game)
            self._current_policy = load_cur_policy
            self._average_policy = load_avg_policy
            self._iteration = load__iterations
            self._initialize_info_state_nodes(self._root_node)

    def save_cfrplus():
        avg_policy = cfr_solver.average_policy()
        avg_policy = dict(zip(avg_policy.state_lookup, avg_policy.action_probability_array))
        policy_handler.save_to_tabular_policy(game, avg_policy, "policies/CFRPlus/{}/{}".format(save_prefix, it))

    cfr_solver = cfr.CFRPlusSolver(game)
    for it in range(load__iterations, load__iterations+iterations+1):  #so that if you tell it to train 20K iterations, the last save isn't 19999
        if save_every != 0 and it%save_every == 0: #order is important
            save_cfrplus()
        cfr_solver.evaluate_and_update_policy()
    save_cfrplus()


def XFP_Solving(game, iterations, save_every = 0, save_prefix = 'temp'):
    def save_xfp():
        xfp_policy = xfp_solver.average_policy_tables()
        policy_keys = np.concatenate((list(xfp_policy[0].keys()), list(xfp_policy[1].keys())), 0)
        policy_values = np.concatenate((list(map(lambda d: list(d.values()), list(xfp_policy[0].values()))),
                                        list(map(lambda d: list(d.values()), list(xfp_policy[1].values())))), 0)
        #change possible None's into 0
        policy_values = [(d if d else 0 for d in a) for a in policy_values]
        xfp_policy = dict(zip(policy_keys, policy_values))
        policy_handler.save_to_tabular_policy(game, xfp_policy, "policies/XFP/{}/{}".format(save_prefix, it))

    xfp_solver = fictitious_play.XFPSolver(game)
    for it in range(iterations+1):
        xfp_solver.iteration()
        if save_every != 0 and it%save_every == 0: #order is important
            save_xfp()
    save_xfp()

#kuhn_policy_gradient.py
def PG_Solving(game, iterations, save_every = 0, save_prefix = 'temp'):
    class PolicyGradientPolicies(policy.Policy):
        """Joint policy to be evaluated."""
        def __init__(self, nfsp_policies):
            player_ids = [0, 1]
            super(PolicyGradientPolicies, self).__init__(game, player_ids)
            self._policies = nfsp_policies
            self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

        def action_probabilities(self, state, player_id=None):
            cur_player = state.current_player()
            legal_actions = state.legal_actions(cur_player)

            self._obs["current_player"] = cur_player
            self._obs["info_state"][cur_player] = (
                state.information_state_tensor(cur_player))
            self._obs["legal_actions"][cur_player] = legal_actions

            info_state = rl_environment.TimeStep(
                observations=self._obs, rewards=None, discounts=None, step_type=None)

            p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
            prob_dict = {action: p[action] for action in legal_actions}
            return prob_dict

    def save_pg():
        tabular_policy = policy.tabular_policy_from_policy(game, expl_policies_avg)
        policy_handler.save_tabular_policy(game, tabular_policy, "policies/PG/{}/{}".format(save_prefix, it))

    num_players = 2
    env = rl_environment.Environment(game, **{"players": num_players})
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    with tf.Session() as sess:
        # pylint: disable=g-complex-comprehension
        agents = [
            policy_gradient.PolicyGradient(
                sess,
                idx,
                info_state_size,
                num_actions,
                loss_str="rpg",         #["rpg", "qpg", "rm"] = PG loss to use.
                hidden_layers_sizes=(128,)) for idx in range(num_players)
        ]
        expl_policies_avg = PolicyGradientPolicies(agents)

        sess.run(tf.global_variables_initializer())
        for it in range(iterations + 1):
            if save_every != 0 and it % save_every == 0:  # order is important
                save_pg()

            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                action_list = [agent_output.action]
                time_step = env.step(action_list)
            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)
        save_pg()


def NFSP_Solving(game, iterations, save_every = 0, save_prefix = 'temp'):
    class NFSPPolicies(policy.Policy):
        """Joint policy to be evaluated."""
        def __init__(self, nfsp_policies, mode):
            player_ids = [0, 1]
            super(NFSPPolicies, self).__init__(game, player_ids)
            self._policies = nfsp_policies
            self._mode = mode
            self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

        def action_probabilities(self, state, player_id=None):
            cur_player = state.current_player()
            legal_actions = state.legal_actions(cur_player)

            self._obs["current_player"] = cur_player
            self._obs["info_state"][cur_player] = (
                state.information_state_tensor(cur_player))
            self._obs["legal_actions"][cur_player] = legal_actions

            info_state = rl_environment.TimeStep(
                observations=self._obs, rewards=None, discounts=None, step_type=None)

            with self._policies[cur_player].temp_mode_as(self._mode):
                p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
            prob_dict = {action: p[action] for action in legal_actions}
            return prob_dict

    def save_nfsp():
        tabular_policy = policy.tabular_policy_from_policy(game, expl_policies_avg)
        policy_handler.save_tabular_policy(game, tabular_policy, "policies/NFSP/{}/{}".format(save_prefix, it))

    num_players = 2
    env_configs = {"players": num_players}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [128]
    replay_buffer_capacity = int(2e5)
    reservoir_buffer_capacity = int(2e6)
    anticipatory_param = 0.1

    hidden_layers_sizes = [int(l) for l in hidden_layers_sizes]
    kwargs = {
        "replay_buffer_capacity": replay_buffer_capacity,
        "epsilon_decay_duration": iterations,
        "epsilon_start": 0.06,
        "epsilon_end": 0.001,
    }

    with tf.Session() as sess:
        # pylint: disable=g-complex-comprehension
        agents = [
            nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                      reservoir_buffer_capacity, anticipatory_param,
                      **kwargs) for idx in range(num_players)
        ]
        expl_policies_avg = NFSPPolicies(agents, nfsp.MODE.average_policy)

        sess.run(tf.global_variables_initializer())
        for it in range(iterations + 1):
            if save_every != 0 and it % save_every == 0:  # order is important
                save_nfsp()

            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                action_list = [agent_output.action]
                time_step = env.step(action_list)
            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)
        save_nfsp()


def print_algorithm_results(game, policy, algorithm_name):
    print(algorithm_name.upper())
    callable_policy = PolicyFromCallable(game, policy)
    policy_exploitability = exploitability(game, callable_policy)
    # print(callable_policy._callable_policy.action_probability_array)
    print("exploitability = {}".format(policy_exploitability))


def deep_CFR_Solving(game, num_iters = 400, num_travers = 40, save_every=0,  save_prefix = 'temp',
                            lr = 1e-3, policy_layers = (32,32), advantage_layers = (16,16)):

    def save_deepcfr():#and print some info i guess?
        print("---------iteration " + str(it) + "----------")
        for player, losses in six.iteritems(advantage_losses):
            print("Advantage for player ", player, losses)
            print("Advantage Buffer Size for player", player,
                  len(deep_cfr_solver.advantage_buffers[player]))
        print("Strategy Buffer Size: ",
              len(deep_cfr_solver.strategy_buffer))
        print("policy loss: ", policy_loss)
        callable_policy = PolicyFromCallable(game, deep_cfr_solver.action_probabilities)
        tabular_policy = tabular_policy_from_policy(game, callable_policy)
        policy = dict(zip(tabular_policy.state_lookup, tabular_policy.action_probability_array))
        # save under map (save_prefix)_(num_travers)
        return policy_handler.save_to_tabular_policy(game, policy, "policies/deepCFR/{}/{}".format(save_prefix + "_" + str(num_travers), it))

    with tf.Session() as sess:
        #set num iters and call solve() multiple times to allow intermediate saving and eval
        deep_cfr_solver = deep_cfr.DeepCFRSolver(sess, game, policy_network_layers=policy_layers,
                                                 advantage_network_layers=advantage_layers, num_iterations=1,
                                                 num_traversals=num_travers, learning_rate=lr)
        sess.run(tf.global_variables_initializer())

        for it in range(num_iters+1):
            _, advantage_losses, policy_loss = deep_cfr_solver.solve()
            if save_every != 0 and it % save_every == 0:
                save_deepcfr()
        return save_deepcfr()

#round policy, can decrease exploitability
def round_tabular_policy_probabilties(policy):
    arr = policy.action_probability_array
    for actions in arr:
        for j,action in enumerate(actions):
            if action > 0.999:
                actions[j] = 1
            if action < 0.001:
                actions[j] = 0
            if 0.667 > action > 0.665:
                actions[j] = 2 / 3
            if 0.334 > action > 0.332:
                actions[j] = 1 / 3
    policy.action_probability_array = arr
    return policy
