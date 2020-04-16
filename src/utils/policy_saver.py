import pickle
import pyspiel
from open_spiel.python.policy import TabularPolicy

#module to handle saving and loading (reconstruct) policies to a binary file using pickle

def save_tabular_policy(game, policy, path):
    dict = {'game': game.get_type().short_name, 'action_probability_array': policy.action_probability_array}
    with open(path, 'wb') as file:
        pickle.dump(dict, file)

def load_tabular_policy(path):
    with open(path, 'rb') as file:
        dict = pickle.load( file)
        game = pyspiel.load_game(dict['game'])
        tabular_policy = TabularPolicy(game)
        tabular_policy.action_probability_array = dict['action_probability_array']
        return tabular_policy