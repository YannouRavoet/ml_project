import pyspiel
import numpy as np
from absl import app
from utils_poker import command_line_action
from open_spiel.python.algorithms import cfr, fictitious_play
from open_spiel.python.algorithms.exploitability import exploitability
from open_spiel.python.policy import TabularPolicy, PolicyFromCallable
import policy_saver

def CFR_Solving(game, iterations):
    cfr_solver = cfr.CFRSolver(game)
    for _ in range(iterations):
        cfr_solver.evaluate_and_update_policy()
    policy = cfr_solver.average_policy()
    return dict(zip(policy.state_lookup, policy.action_probability_array))


def XFP_Solving(game, iterations):
    xfp_solver = fictitious_play.XFPSolver(game)
    for _ in range(iterations):
        xfp_solver.iteration()
    policy = xfp_solver.average_policy_tables()
    policy_keys = np.concatenate((list(policy[0].keys()), list(policy[1].keys())), 0)
    policy_values = np.concatenate((list(map(lambda d: [d.get(0), d.get(1)], list(policy[0].values()))),
                                    list(map(lambda d: [d.get(0), d.get(1)], list(policy[1].values())))), 0)
    return dict(zip(policy_keys, policy_values))


def print_algorithm_results(game, policy, algorithm_name):
    print(algorithm_name.upper())
    callable_policy = PolicyFromCallable(game, policy)
    policy_exploitability = exploitability(game, callable_policy)
    # print(callable_policy._callable_policy.action_probability_array)
    print("exploitability = {}".format(policy_exploitability))


def main(_):
    n = 100;
    game = pyspiel.load_game("kuhn_poker")  # kuhn_poker or leduc_poker
    tabular_policy = TabularPolicy(game)
    state_lookup_order = list(tabular_policy.state_lookup.keys())

    # CFR
    cfr_policy = CFR_Solving(game, iterations=n)
    # order the policy values based on tabular_policy order
    cfr_policy = {k: cfr_policy.get(k) for k in state_lookup_order}
    tabular_policy.action_probability_array = list(cfr_policy.values())
    print_algorithm_results(game, tabular_policy, "cfr")

    #example: save, reload and test the policy again
    policy_saver.save_tabular_policy(game, tabular_policy, "policies/CFR1")
    loaded_policy = policy_saver.load_tabular_policy("policies/CFR1")
    print_algorithm_results(game, loaded_policy, "cfr_reloaded")

    # XFP
    xfp_policy = XFP_Solving(game, iterations=n)
    # order the policy values based on tabular_policy order
    xfp_policy = {k: xfp_policy.get(k) for k in state_lookup_order}
    tabular_policy.action_probability_array = list(xfp_policy.values())
    print_algorithm_results(game, tabular_policy, "xfp")

    #example: save, reload and test the policy again
    policy_saver.save_tabular_policy(game, tabular_policy, "policies/XFP1")
    loaded_policy = policy_saver.load_tabular_policy("policies/XFP1")
    print_algorithm_results(game, loaded_policy, "xfp_reloaded")

if __name__ == "__main__":
    app.run(main)
