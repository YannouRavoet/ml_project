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
    # laat ze num_episodes games tegen elkaar spelen en returned win % van eerste policy
    # zou ook moeten werken voor leduc dus misschien verplaatsen naar utils
    wins = [0,0]

    for i in range(num_episodes):
        state = game.new_initial_state()
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
                # niet zeker of dit correct is, niet gebruikt in kuhn poker denk ik
                chosen_actions = [
                    np.random.choice(state.legal_actions(pid), p=list(pid.action_probabilities(state).values()))
                    for pid in policies
                ]
                state.apply_actions(chosen_actions)

            else:
                # Decision node: sample action for the single current player
                action = np.random.choice(state.legal_actions(state.current_player()), p=list(policies[state.current_player()].action_probabilities(state).values()))
                state.apply_action(action)

            print(str(state))

        # Game is now done. increment win of winner
        returns = state.returns()
        if returns[0] > returns[1]:
            wins[0] += 1
        else:
            wins[1] += 1

    print("player wins matrix: " + str(wins))
    win_percentage = wins[0]/(wins[0] + wins[1])
    print("player 0 wins: " + str(win_percentage))
    return win_percentage

def main(_):
    n = 10000;
    game = pyspiel.load_game("kuhn_poker")  # kuhn_poker or leduc_poker
    tabular_policy = TabularPolicy(game)
    state_lookup_order = list(tabular_policy.state_lookup.keys())

    # CFR
    cfr_policy = CFR_Solving(game, iterations=n)
    # order the policy values based on tabular_policy order
    cfr_policy = {k: cfr_policy.get(k) for k in state_lookup_order}
    tabular_policy.action_probability_array = list(cfr_policy.values())
    print_algorithm_results(game, tabular_policy, "cfr")

    #example: save, reload and test the policy again
    policy_saver.save_tabular_policy(game, tabular_policy, "policies/CFR1")
    loaded_policy1 = policy_saver.load_tabular_policy("policies/CFR1")
    print_algorithm_results(game, loaded_policy1, "cfr_reloaded")

    # XFP
    xfp_policy = XFP_Solving(game, iterations=n)
    # order the policy values based on tabular_policy order
    xfp_policy = {k: xfp_policy.get(k) for k in state_lookup_order}
    tabular_policy.action_probability_array = list(xfp_policy.values())
    print_algorithm_results(game, tabular_policy, "xfp")

    #example: save, reload and test the policy again
    policy_saver.save_tabular_policy(game, tabular_policy, "policies/XFP1")
    loaded_policy2 = policy_saver.load_tabular_policy("policies/XFP1")
    print_algorithm_results(game, loaded_policy2, "xfp_reloaded")

    eval_against_policy(game, [loaded_policy1,loaded_policy2],1000)

if __name__ == "__main__":
    app.run(main)
