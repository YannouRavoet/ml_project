import pyspiel
import numpy as np
from absl import app
from utils_poker import command_line_action
from open_spiel.python.algorithms import cfr, fictitious_play
from open_spiel.python.algorithms.exploitability import exploitability
from open_spiel.python.policy import TabularPolicy, PolicyFromCallable
from open_spiel.python.algorithms import generate_playthrough
import policy_saver

def CFR_Solving(game, iterations):
    cfr_solver = cfr.CFRSolver(game)
    for _ in range(iterations):
        cfr_solver.evaluate_and_update_policy()
    policy = cfr_solver.average_policy()
    return dict(zip(policy.state_lookup, policy.action_probability_array))


def XFP_Solving(game, iterations):
    xfp_solver = fictitious_play.XFPSolver(game)
    for _ in range(iterations):
        xfp_solver.iteration()
    policy = xfp_solver.average_policy_tables()
    policy_keys = np.concatenate((list(policy[0].keys()), list(policy[1].keys())), 0)
    policy_values = np.concatenate((list(map(lambda d: [d.get(0), d.get(1)], list(policy[0].values()))),
                                    list(map(lambda d: [d.get(0), d.get(1)], list(policy[1].values())))), 0)
    return dict(zip(policy_keys, policy_values))


def print_algorithm_results(game, policy, algorithm_name):
    print(algorithm_name.upper())
    callable_policy = PolicyFromCallable(game, policy)
    policy_exploitability = exploitability(game, callable_policy)
    # print(callable_policy._callable_policy.action_probability_array)
    print("exploitability = {}".format(policy_exploitability))

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


def main(_):
    n = 1;
    game = pyspiel.load_game("kuhn_poker")  # kuhn_poker or leduc_poker
    tabular_policy1 = TabularPolicy(game)
    tabular_policy2 = TabularPolicy(game)
    state_lookup_order = list(tabular_policy1.state_lookup.keys())

    # CFR
    cfr_policy = CFR_Solving(game, iterations=n)
    # order the policy values based on tabular_policy order
    cfr_policy = {k: cfr_policy.get(k) for k in state_lookup_order}
    tabular_policy1.action_probability_array = list(list(val) for val in cfr_policy.values())
    print_algorithm_results(game, tabular_policy1, "cfr")


    #save policy
    #policy_saver.save_tabular_policy(game, tabular_policy1, "policies/CFR10k")

    # XFP
    xfp_policy = XFP_Solving(game, iterations=n)
    # order the policy values based on tabular_policy order
    xfp_policy = {k: xfp_policy.get(k) for k in state_lookup_order}
    tabular_policy2.action_probability_array = list(list(val) for val in xfp_policy.values())
    print_algorithm_results(game, tabular_policy2, "xfp")

    #save policy
    #policy_saver.save_tabular_policy(game, tabular_policy2, "policies/XFP10k")


    #load enkele policies, "10k" staat voor aantal iteraties getraind
    CFR10 = policy_saver.load_tabular_policy("policies/CFR10k")
    CFR100 = policy_saver.load_tabular_policy("policies/CFR100k")
    #XFP10 = policy_saver.load_tabular_policy("policies/XFP10k")


    print("cfr100 vs cfr10")
    #todo: dit zou de tweede keer het omgekeerde van de eerste keer moeten returnen, maar is zelfs niet het geval bij 1e6 iters...
    eval_against_policy(game, [CFR100,CFR10], 10000)
    eval_against_policy(game, [CFR10, CFR100], 10000)





if __name__ == "__main__":
    app.run(main)
