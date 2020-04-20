import pyspiel
import numpy as np
from absl import app
from open_spiel.python.algorithms import cfr, fictitious_play
from open_spiel.python.algorithms.exploitability import exploitability, best_response
from open_spiel.python.policy import TabularPolicy, PolicyFromCallable
from open_spiel.python.algorithms import generate_playthrough
import policy_handler
import utils_poker

def main(_):
    n = 1
    game = pyspiel.load_game("leduc_poker")
    tabular_policy1 = TabularPolicy(game)
    tabular_policy2 = TabularPolicy(game)
    state_lookup_order = list(tabular_policy1.state_lookup.keys())


    # CFR
    cfr_policy = utils_poker.CFR_Solving(game, iterations=n)

    utils_poker.print_algorithm_results(game, cfr_policy, "cfr")

    #save policy
    #policy_handler.save_tabular_policy(game, tabular_policy1, "policies/CFR100")

    # XFP
    xfp_policy = utils_poker.XFP_Solving(game, iterations=n)
    utils_poker.print_algorithm_results(game, xfp_policy, "xfp")

    #save policy
    #policy_handler.save_tabular_policy(game, tabular_policy2, "policies/XFP100")

    CFR100 = policy_handler.load_to_tabular_policy("policies/CFR100")
    XFP100 = policy_handler.load_to_tabular_policy("policies/XFP100")

    utils_poker.print_algorithm_results(game,CFR100, "CFR100")
    utils_poker.print_algorithm_results(game, XFP100, "XFP100")


    print("CFR100 vs XFP100:")
    policy_handler.eval_against_policy(game, [CFR100, XFP100], 10000)
    print("XFP100 vs CFR100")
    policy_handler.eval_against_policy(game, [XFP100, CFR100], 10000)



if __name__ == "__main__":
    app.run(main)
