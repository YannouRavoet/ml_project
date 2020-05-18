import pyspiel
from absl import app
import utils_poker
import policy_handler
from ML_project20 import tournament

def train_policies(game, iterations=0):
    utils_poker.CFR_Solving(game, iterations=iterations, save_every=1000, save_prefix='base')
    # utils_poker.XFP_Solving(game, iterations=iterations, save_every=1000, save_prefix='base')
    # utils_poker.CFRPlus_Solving(game, iterations=iterations, save_every=1000, save_prefix='base')
    # utils_poker.PG_Solving(game, iterations=iterations, save_every=10000, save_prefix='base')
    # utils_poker.NFSP_Solving(game, iterations=iterations, save_every=10000, save_prefix='base')
    # utils_poker.DCFR_Solving(game, iterations=iterations, save_every=1000, save_prefix='g=1', g=1)
    return

def main(_):
    game = pyspiel.load_game("kuhn_poker")  # kuhn_poker or leduc_poker

    # # TRAINING
    # n = int(1e6)
    # train_policies(game, n)

    # TESTING
    # PLOTTING
    utils_poker.plot_policies(game, {'CFR': 'CFR/base/', 'XFP': 'XFP/base/', 'PG': 'PG/base/'}, extract_metrics=False, max_iter=int(1e6))
    # ROUNDING
    # cfr_policy = policy_handler.load_to_tabular_policy('policies/CFRPlus/base/1000000')
    # utils_poker.print_algorithm_results(game, cfr_policy, 'CFRPlus no rounding')
    # cfr_rounded_policy = utils_poker.round_tabular_policy_probabilties(cfr_policy)
    # utils_poker.print_algorithm_results(game, cfr_rounded_policy, 'CFRPlus with rounding')

    # AGAINST RANDOM BOTS
    # random_policy = policy_handler.load_to_tabular_policy('policies/CFRPlus/base/0')
    # utils_poker.eval_against_policy(game, {'CFR+': cfr_policy, 'Random': random_policy}, num_episodes=10, num_iterations=int(1e4))


    # CSV FILE
    # tournament.policy_to_csv(game, cfr_policy, 'kuhn_cfrplus_1M.csv')

if __name__ == "__main__":
    app.run(main)
