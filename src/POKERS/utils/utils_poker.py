import os
import sys
import six
import pickle
import numpy as np
import policy_handler
import tensorflow as tf
import matplotlib.pyplot as plt

from open_spiel.python.policy import tabular_policy_from_callable
from open_spiel.python.algorithms.exploitability import exploitability, nash_conv
from open_spiel.python import policy
from open_spiel.python.algorithms import cfr, fictitious_play, policy_gradient, nfsp, discounted_cfr, deep_cfr, cfr_br
from open_spiel.python import rl_environment


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


""" PLOTTING """


def plot_policies(game, algorithms, extract_metrics=True, max_iter = None):
    """
    :param game: pyspiel Game class
    :param algorithms: {string: string} maps the algorithm name to the prefix within the policies directory of the game directory (i.e. {'CFR': 'base/'})
    :param extract_metrics: bool - True if you still have to extract the metrics from the saved policies
    :param max_iter: int - max iteration to plot
    :return: void
    """
    exploitabilities = {}
    nash_convs = {}
    for algo in algorithms:
        print(str(algo).upper())
        algo_path = algorithms[algo]
        if extract_metrics:
            files = get_algo_files(algo_path, max_iter)
            algo_policies = get_algo_policies(algo_path, files, game)
            algo_exploitabilities, algo_nashconvs = get_algo_metrics(algo_policies, game)
            save_metrics(algo_exploitabilities, algo_nashconvs, algo_path)
        else:
            algo_exploitabilities, algo_nashconvs = load_metrics(algo_path)

        exploitabilities[algo] = algo_exploitabilities
        nash_convs[algo] = algo_nashconvs

    # PLOTTING
    print("Plotting metrics for all passed algorithms...")
    plot_series('Exploitability ifo training iterations', 'Exploitability', exploitabilities, max_iter)
    plot_series('NashConv ifo training iterations', 'NashConv', nash_convs, max_iter)
    return


def get_algo_files(algo_path, max_iter):
    print("Getting the files...")
    files = np.array([])
    for (root, subFolder, filenames) in os.walk('policies'):
        for file in filenames:
            path = os.path.join(root, file)
            if algo_path in path and (max_iter is None or int(file) <= max_iter):
                files = np.append(files, path)
    return files


def get_algo_policies(algo_path, files, game):
    print("Extracting the policies...")
    algo_policies = {}
    for file in files:
        algo_iterations = (int(file.split(algo_path)[1]))
        algo_policy = policy_handler.load_to_tabular_policy(file)
        algo_policies[algo_iterations] = policy.tabular_policy_from_callable(game, algo_policy)
    return algo_policies


def get_algo_metrics(algo_policies, game):
    print("Extracting metrics...")
    algo_exploitabilities = {}
    algo_nashconvs = {}
    for key in algo_policies:
        algo_exploitabilities[key] = exploitability(game, algo_policies[key])
        algo_nashconvs[key] = nash_conv(game, algo_policies[key])
    return algo_exploitabilities, algo_nashconvs


def save_metrics(algo_exploitabilities, algo_nashconvs, algo_path):
    print("Saving metrics...")
    algo_expl_path = os.path.join('metrics', algo_path, 'exploitabilities')
    algo_nashconv_path = os.path.join('metrics', algo_path, 'nashconv')
    with open(algo_expl_path, 'wb') as file:
        pickle.dump(algo_exploitabilities, file)
    with open(algo_nashconv_path, 'wb') as file:
        pickle.dump(algo_nashconvs, file)


def load_metrics(algo_path):
    print("Loading metrics...")
    algo_expl_path = os.path.join('metrics', algo_path, 'exploitabilities')
    algo_nashconv_path = os.path.join('metrics', algo_path, 'nashconv')
    with open(algo_expl_path, 'rb') as file:
        algo_exploitabilities = pickle.load(file)
    with open(algo_nashconv_path, 'rb') as file:
        algo_nashconvs = pickle.load(file)
    return algo_exploitabilities, algo_nashconvs


def plot_series(title, metric, series, max_iter):
    legend = []
    for algo in series:
        algo_series = series[algo]
        algo_series = sorted(algo_series.items())
        if not max_iter is None:
            algo_series = list(filter(lambda x: x[0]<=max_iter, algo_series))
        algo_iterations, algo_series = zip(*algo_series)
        plt.plot(algo_iterations, algo_series)
        legend.append(algo)

    plt.title(title)
    plt.legend(legend)
    plt.xlabel('number of training iterations')
    plt.ylabel(metric)
    plt.yscale('log')
    plt.show()


def print_algorithm_results(game, callable_policy, algorithm_name):
    print(algorithm_name.upper())
    tabular_policy = tabular_policy_from_callable(game, callable_policy)
    policy_exploitability = exploitability(game, tabular_policy)
    policy_nashconv = nash_conv(game, tabular_policy)
    print("exploitability = {}".format(policy_exploitability))
    print("nashconv = {}".format(policy_nashconv))


""" TRAINING ALGORITHMS """


def CFR_Solving(game, iterations, save_every=0, save_prefix='base'):
    def save_cfr():
        policy = cfr_solver.average_policy()
        policy = dict(zip(policy.state_lookup, policy.action_probability_array))
        policy_handler.save_to_tabular_policy(game, policy, "policies/CFR/{}/{}".format(save_prefix, it))

    cfr_solver = cfr.CFRSolver(game)
    for it in range(iterations + 1):
        if save_every != 0 and it % save_every == 0:  # order is important
            save_cfr()
        cfr_solver.evaluate_and_update_policy()
    save_cfr()


def DCFR_Solving(game, iterations, save_every=0, save_prefix='base', a=3 / 2, b=0, g=2):
    def save_dcfr():
        policy = dcfr_solver.average_policy()
        policy = dict(zip(policy.state_lookup, policy.action_probability_array))
        policy_handler.save_to_tabular_policy(game, policy, "policies/CFR_Discounted/{}/{}".format(save_prefix, it))

    dcfr_solver = discounted_cfr.DCFRSolver(game, alpha=a, beta=b, gamma=g)
    for it in range(iterations + 1):
        if save_every != 0 and it % save_every == 0:  # order is important
            save_dcfr()
        dcfr_solver.evaluate_and_update_policy()
    save_dcfr()


def CFR_BR_Solving(game, iterations, save_every=0, save_prefix='base'):
    def save_cfr_br():
        policy = cfr_solver.average_policy()
        policy = dict(zip(policy.state_lookup, policy.action_probability_array))
        policy_handler.save_to_tabular_policy(game, policy, "policies/CFRBR/{}/{}".format(save_prefix, it))

    cfr_solver = cfr_br.CFRBRSolver(game)
    for it in range(iterations + 1):
        if save_every != 0 and it % save_every == 0:  # order is important
            save_cfr_br()
        cfr_solver.evaluate_and_update_policy()
    save_cfr_br()


def CFRPlus_Solving(game, iterations, save_every=0, save_prefix='base', alternating_updates = True, linear_averaging = True):
    def save_cfrplus():
        avg_policy = cfr_solver.average_policy()
        avg_policy = dict(zip(avg_policy.state_lookup, avg_policy.action_probability_array))
        policy_handler.save_to_tabular_policy(game, avg_policy, "policies/CFRPlus/{}/{}".format(save_prefix, it))

    cfr_solver = cfr.CFRPlusSolver(game)
    #cfr_solver = cfr._CFRSolver(game, regret_matching_plus=True, alternating_updates=alternating_updates, linear_averaging=linear_averaging)
    for it in range(iterations + 1):  # so that if you tell it to train 20K iterations, the last save isn't 19999

        if save_every != 0 and it % save_every == 0:  # order is important
            save_cfrplus()
        cfr_solver.evaluate_and_update_policy()
    save_cfrplus()


def XFP_Solving(game, iterations, save_every=0, save_prefix='base'):
    def save_xfp():
        xfp_policy = xfp_solver.average_policy_tables()
        policy_keys = np.concatenate((list(xfp_policy[0].keys()), list(xfp_policy[1].keys())), 0)
        policy_values = np.concatenate((list(map(lambda d: list(d.values()), list(xfp_policy[0].values()))),
                                        list(map(lambda d: list(d.values()), list(xfp_policy[1].values())))), 0)
        # change possible None's into 0
        policy_values = [(d if d else 0 for d in a) for a in policy_values]
        xfp_policy = dict(zip(policy_keys, policy_values))
        policy_handler.save_to_tabular_policy(game, xfp_policy, "policies/XFP/{}/{}".format(save_prefix, it))

    xfp_solver = fictitious_play.XFPSolver(game)
    for it in range(iterations + 1):
        xfp_solver.iteration()
        if save_every != 0 and it % save_every == 0:  # order is important
            save_xfp()
    save_xfp()


# kuhn_policy_gradient.py
def PG_Solving(game, iterations, save_every=0, save_prefix='base'):
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
        tabular_policy = policy.tabular_policy_from_callable(game, expl_policies_avg)
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
                loss_str="rpg",  # ["rpg", "qpg", "rm"] = PG loss to use.
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


def NFSP_Solving(game, iterations, save_every=0, save_prefix='base'):
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
        tabular_policy = policy.tabular_policy_from_callable(game, expl_policies_avg)
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


def DEEPCFR_Solving(game, iterations, save_every=0, save_prefix='base', num_travers=40,
                    lr=1e-3, policy_layers=(32, 32), advantage_layers=(16, 16)):
    def save_deepcfr():  # and print some info i guess?
        print("---------iteration " + str(it) + "----------")
        for player, losses in six.iteritems(advantage_losses):
            print("Advantage for player ", player, losses)
            print("Advantage Buffer Size for player", player,
                  len(deep_cfr_solver.advantage_buffers[player]))
        print("Strategy Buffer Size: ",
              len(deep_cfr_solver.strategy_buffer))
        print("policy loss: ", policy_loss)
        callable_policy = tabular_policy_from_callable(game, deep_cfr_solver.action_probabilities)
        tabular_policy = tabular_policy_from_callable(game, callable_policy)
        policy = dict(zip(tabular_policy.state_lookup, tabular_policy.action_probability_array))
        # save under map (save_prefix)_(num_travers)
        return policy_handler.save_to_tabular_policy(game, policy, "policies/DEEPCFR/{}/{}".format(
            save_prefix + "_" + str(num_travers), it))

    with tf.Session() as sess:
        # set num iters and call solve() multiple times to allow intermediate saving and eval
        deep_cfr_solver = deep_cfr.DeepCFRSolver(sess, game, policy_network_layers=policy_layers,
                                                 advantage_network_layers=advantage_layers, num_iterations=1,
                                                 num_traversals=num_travers, learning_rate=lr)
        sess.run(tf.global_variables_initializer())

        for it in range(iterations + 1):
            _, advantage_losses, policy_loss = deep_cfr_solver.solve()
            if save_every != 0 and it % save_every == 0:
                save_deepcfr()
        return save_deepcfr()


""" ADAPTING POLICIES """


# round policy, can decrease exploitability
def round_tabular_policy_probabilties(policy, vals=[1, 0, 1 / 3, 2 / 3], th=0.001):
    arr = policy.action_probability_array
    for actions in arr:
        # array indicating which actions were rounded to detect if some but not all were rounded
        rounded = np.zeros(len(actions), dtype=bool)
        for j, action in enumerate(actions):
            for val in vals:
                if abs(action - val) < th:
                    actions[j] = val
                    rounded[j] = True

            #if some but not all were rounded, we need to rescale the not-rounded ones to ensure sum = 1
            if rounded.any() and not rounded.all():
                #get indices of the actions that were not rounded
                unrounded = np.where(rounded==False)[0]
                #calculate residual to spread over unrounded vals
                delta = (1 - sum(actions))/len(unrounded)
                for index in unrounded:
                    actions[index] += delta

    policy.action_probability_array = arr
    return policy
