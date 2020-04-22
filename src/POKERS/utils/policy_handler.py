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


def eval_against_policy(game, policies, num_episodes):
    # input: game, 2 policies in een array, aantal games
    # laat ze num_episodes games tegen elkaar spelen en return de soms van utilities
    # zou ook moeten werken voor leduc dus misschien verplaatsen naar utils
    returns = [0,0]

    for i in range(num_episodes):
        state = game.new_initial_state()
        dealt_cards = []
        decisions = []
        #order veranderen elke iteratie (wie eerst mag)
        order = i%2
        while not state.is_terminal():
            # The state can be two different types: chance node or decision node
            if state.is_chance_node():
                # DEAL CARDS:
                # kuhn_poker POKER => Player 0 = 33%/card, Player 1 = 50%/card
                outcomes = state.chance_outcomes()
                card_list, prob_list = zip(*outcomes)
                card = np.random.choice(card_list, p=prob_list)
                dealt_cards.append(state.action_to_string(card))
                state.apply_action(card)

            else:
                # Decision node: sample decision for the current player
                # change used policy adhv order:
                used_policy = abs(state.current_player() - order)
                decision = np.random.choice(state.legal_actions(state.current_player()), p=list(policies[used_policy].action_probabilities(state).values()))
                decisions.append(state.action_to_string(decision))
                state.apply_action(decision)



        # Game is now done. add utilities to correct policy
        returns[0] += state.returns()[abs(0-order)]
        returns[1] += state.returns()[abs(1-order)]

    print("policy utility matrix: " + str(returns))
    return returns
