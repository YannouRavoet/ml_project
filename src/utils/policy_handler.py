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
        #order veranderen elke iteratie (wie eerst mag)
        order = i%2
        while not state.is_terminal():
            # The state can be three different types: chance node,
            # simultaneous node, or decision node
            if state.is_chance_node():
                # Chance node: sample an outcome
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)


            elif state.is_simultaneous_node():
                # Simultaneous node: sample actions for all players.
                # todo: niet zeker of dit correct is, niet gebruikt in kuhn poker!!
                chosen_actions = [
                    np.random.choice(state.legal_actions(pid), p=list(pid.action_probabilities(state).values()))
                    for pid in policies
                ]
                if(order):
                    chosen_actions = reversed(chosen_actions)
                state.apply_actions(chosen_actions)

            else:
                # Decision node: sample action for the single current player
                # change player index adhv order:
                player = abs(state.current_player() - order)
                action = np.random.choice(state.legal_actions(state.current_player()), p=list(policies[player].action_probabilities(state).values()))
                state.apply_action(action)



        # Game is now done. add utilities to correct player
        returns[0] += state.returns()[abs(0-order)]
        returns[1] += state.returns()[abs(1-order)]

    print("player utility matrix: " + str(returns))

    return returns
