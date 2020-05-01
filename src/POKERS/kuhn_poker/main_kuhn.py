import pyspiel
from absl import app
import utils_poker

def train_policies(game, iterations=0):
    utils_poker.CFR_Solving(game, iterations=iterations, save_every=1000, save_prefix='temp')
    # utils_poker.XFP_Solving(game, iterations=iterations, save_every=1000, save_prefix='temp')
    # utils_poker.CFRPlus_Solving(game, iterations=iterations, save_every=1000, save_prefix='temp')
    # utils_poker.PG_Solving(game, iterations=iterations, save_every=10000, save_prefix='temp')
    # utils_poker.NFSP_Solving(game, iterations=iterations, save_every=10000, save_prefix='temp')
    # utils_poker.DCFR_Solving(game, iterations=iterations, save_every=1000, save_prefix='temp')
    return

def main(_):
    game = pyspiel.load_game("kuhn_poker")  # kuhn_poker or leduc_poker
    # TRAINING
    n = int(5e6)
    # train_policies(game, n)

    # TESTING
    utils_poker.plot_policies(game, {'CFR': 'temp/'}, extract_metrics=True, max_iter=int(5e6)) #, 'XFP': 'temp/', 'PG': 'temp/', 'CFRPlus': 'temp/', 'DCFR': 'temp/', 'NFSP': 'temp/'


if __name__ == "__main__":
    app.run(main)
