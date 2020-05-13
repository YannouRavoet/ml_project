import pickle
import pyspiel
import numpy as np
from open_spiel.python.policy import TabularPolicy
from datetime import datetime

#module to handle saving and loading (reconstruct) policies to a binary file using pickle
#todo: directory must exist when saving
def save_to_tabular_policy(game, policy, path):
    # MAKING SURE THE POLICY IS SAVED AND LOADED IN THE SAME ORDER
    tabular_policy = TabularPolicy(game)                                                        #create default policy for the game
    state_lookup_order = list(tabular_policy.state_lookup.keys())                               #save the default state_lookup order
    policy = {k: policy.get(k) for k in state_lookup_order}                                     #set the action_probs from the given policy in the correct order
    tabular_policy.action_probability_array = list(list(val) for val in policy.values())        #save the action probs to the default policy
    save_tabular_policy(game, tabular_policy, path)

def save_tabular_policy(game, tabular_policy, path):
    # ACTUAL SAVING OF INFORMATION IN THE POLICY
    dict = {'game': game.get_type().short_name, 'action_probability_array': tabular_policy.action_probability_array}
    with open(path, 'wb') as file:
        pickle.dump(dict, file)
    print("{}: {} saved...".format(datetime.now().strftime("%H:%M:%S"), path))
    return tabular_policy

def load_to_tabular_policy(path):
    with open(path, 'rb') as file:
        dict = pickle.load( file)
        game = pyspiel.load_game(dict['game'])
        tabular_policy = TabularPolicy(game)            #This means that at saving the action probabilities must be ordered in the correct way.
        tabular_policy.action_probability_array = dict['action_probability_array']
        return tabular_policy

