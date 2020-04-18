import pickle
import pyspiel
import numpy as np
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
                # KUHN POKER => Player 0 = 33%/card, Player 1 = 50%/card
                outcomes = state.chance_outcomes()
                card_list, prob_list = zip(*outcomes)
                card = np.random.choice(card_list, p=prob_list)
                dealt_cards.append(state.action_to_string(card))
                state.apply_action(card)


            # elif state.is_simultaneous_node():
            #     # Simultaneous node: sample actions for all players.
            #     # todo: niet zeker of dit correct is, niet gebruikt in kuhn poker!!
            #     # todo: dit is inderdaad niet van toepassing in turn based games
            #     chosen_actions = [
            #         np.random.choice(state.legal_actions(pid), p=list(pid.action_probabilities(state).values()))
            #         for pid in policies
            #     ]
            #     if(order):
            #         chosen_actions = reversed(chosen_actions)
            #     state.apply_actions(chosen_actions)

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
