import pyspiel
import numpy as np
from absl import app
import utils_poker
from open_spiel.python.algorithms import cfr, fictitious_play
import policy_handler


def train_policies(game, iterations=0):
    utils_poker.CFR_Solving(game, iterations=iterations, save_every=1000, save_prefix='temp')
    utils_poker.XFP_Solving(game, iterations=iterations, save_every=1000, save_prefix='temp')
    utils_poker.CFRPlus_Solving(game, iterations=iterations, save_every=1000, save_prefix='temp')


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
