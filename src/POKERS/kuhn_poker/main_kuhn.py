import pyspiel
import numpy as np
from absl import app
import utils_poker
from open_spiel.python.algorithms import cfr, fictitious_play
from open_spiel.python.algorithms.exploitability import exploitability, best_response
from open_spiel.python.policy import TabularPolicy, PolicyFromCallable
from open_spiel.python.algorithms import generate_playthrough
import policy_handler

def CFR_Solving(game, iterations, save_every = 0, save_prefix = 'temp'):
    def save_cfr():
        policy = cfr_solver.average_policy()
        policy = dict(zip(policy.state_lookup, policy.action_probability_array))
        return policy_handler.save_to_tabular_policy(game, policy, "policies/CFR/{}/{}_{}".format(save_prefix, save_prefix, it))

    cfr_solver = cfr.CFRSolver(game)
    for it in range(iterations+1):  #so that if you tell it to train 20K iterations, the last save isn't 19999
        cfr_solver.evaluate_and_update_policy()
        if save_every != 0 and it%save_every == 0: #order is important
            save_cfr()
    return save_cfr()


def XFP_Solving(game, iterations, save_every = 0, save_prefix = 'temp'):
    def save_xfp():
        policy = xfp_solver.average_policy_tables()
        policy_keys = np.concatenate((list(policy[0].keys()), list(policy[1].keys())), 0)
        policy_values = np.concatenate((list(map(lambda d: [d.get(0), d.get(1)], list(policy[0].values()))),
                                        list(map(lambda d: [d.get(0), d.get(1)], list(policy[1].values())))), 0)
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


def main(_):
    n = int(20e3)
    game = pyspiel.load_game("kuhn_poker")  # kuhn_poker or leduc_poker

    # # # TRAINING
    # # CFR
    # cfr_policy = CFR_Solving(game, iterations=n, save_every=1000)

    # # XFP
    # xfp_policy = XFP_Solving(game, iterations=n)

    utils_poker.plot_policies(game, 'CFR/temp/temp_')


    # # TESTING
    # #load enkele policies, "10k" staat voor aantal iteraties getraind
    # CFR1e6 = policy_handler.load_to_tabular_policy("policies/CFR1M")
    # CFR1e5 = policy_handler.load_to_tabular_policy("policies/CFR100k")
    # #XFP1e4 = policy_handler.load_tabular_policy("policies/XFP10k")
    #
    #
    # print("cfr100K vs cfr1M")
    # #todo: CFR 10 en 100 zijn bijna identiek, dat is de reden dat dit totaal niet consistent is denk ik?
    # policy_handler.eval_against_policy(game, [CFR1e5,CFR1e6], 10000)
    # # policy_handler.eval_against_policy(game, [CFR10, tabular_policy1], 10000) # orde dat je policies meegeeft maakt niet uit (wordt intern verhandelt)





if __name__ == "__main__":
    app.run(main)
