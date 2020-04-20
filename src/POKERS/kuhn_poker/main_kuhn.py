import pyspiel
import numpy as np
from absl import app
import utils_poker
from open_spiel.python.algorithms import cfr, fictitious_play
import policy_handler

"""TRAINING"""
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


def train_policies(game, iterations=0):
    CFR_Solving(game, iterations=iterations, save_every=1000, save_prefix='temp')
    XFP_Solving(game, iterations=iterations, save_every=1000, save_prefix='temp')
    CFRPlus_Solving(game, iterations=iterations, save_every=1000, save_prefix='temp')




def main(_):
    n = int(100e3)
    game = pyspiel.load_game("kuhn_poker")  # kuhn_poker or leduc_poker

    # TRAINING
    train_policies(game, n)

    # TESTING
    utils_poker.plot_policies(game, {'CFR':'CFR/temp/', 'XFP':'XFP/temp/', 'CFR+':'CFRPlus/temp/'})

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
