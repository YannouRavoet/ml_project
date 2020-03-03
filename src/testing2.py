import pyspiel
import random
import logging
import numpy as np
from absl import app
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner

#open_spiel/python/examples/matrix_game_example.py
def _matching_pennies():
    game_type = pyspiel.GameType(
        "matching_pennies",
        "Matching Pennies",
        pyspiel.GameType.Dynamics.SIMULTANEOUS,
        pyspiel.GameType.ChanceMode.DETERMINISTIC,
        pyspiel.GameType.Information.ONE_SHOT,
        pyspiel.GameType.Utility.ZERO_SUM,
        pyspiel.GameType.RewardModel.TERMINAL,
        2,  # max num players
        2,  # min_num_players
        True,  # provides_information_state
        True,  # provides_information_state_tensor
        False,  # provides_observation
        False,  # provides_observation_tensor
        dict()  # parameter_specification
    )
    game = pyspiel.MatrixGame(
        game_type,
        {},  # game_parameters
        ["Heads", "Tails"],  # row_action_names
        ["Heads", "Tails"],  # col_action_names                     Heads   Tails
        [[-1, 1], [1, -1]],  # row player utilities         Heads   -1,1    1,-1
        [[1, -1], [-1, 1]]  # col player utilities          Tails   1,-1    -1,1
    )
    return game

def _prisoners_dilemma():
    game_type = pyspiel.GameType(
        "prisonners_dilemma",
        "Prisoners Dilemma",
        pyspiel.GameType.Dynamics.SIMULTANEOUS,
        pyspiel.GameType.ChanceMode.DETERMINISTIC,
        pyspiel.GameType.Information.ONE_SHOT,
        pyspiel.GameType.Utility.NON_ZERO_SUM,              #Non-Zero SUM Game
        pyspiel.GameType.RewardModel.TERMINAL,
        2,  # max num players
        2,  # min_num_players
        True,  # provides_information_state
        True,  # provides_information_state_tensor
        False,  # provides_observation
        False,  # provides_observation_tensor
        dict()  # parameter_specification
    )
    game = pyspiel.MatrixGame(
        game_type,
        {},  # game_parameters
        ["Talk", "Silent"],  # row_action_names
        ["Talk", "Silent"],  # col_action_names                     Talk    Silent
        [[-6, 0], [-12, -3]],  # row player utilities       Talk    -6,-6   0,-12
        [[-6, -12], [0, -3]]  # col player utilities        Silent  -12,0   -3,-3
    )
    return game

def _matching_pennies_easy():
    return pyspiel.create_matrix_game("matching_pennies", "Matching Pennies",                           #        Heads   Tails
                                    ["Heads", "Tails"], ["Heads", "Tails"],                             #Heads   -1,1    1,-1
                                    [[-1, 1], [1, -1]],                                                 #Tails   1,-1    -1,1
                                    [[1, -1], [-1, 1]])

def _prisonners_dilemma_easy():
    return pyspiel.create_matrix_game("prisonners_dilemma", "Prisoners Dilemma",                        #        Talk    Silent
                                    ["Talk", "Silent"], ["Talk", "Silent"],                             #Talk    -6,-6   0,-12
                                    [[-6, 0], [-12, -3]],                                               #Silent  -12,0   -3,-3
                                    [[-6, -12], [0, -3]])

def _battle_of_the_sexes_easy():
    return pyspiel.create_matrix_game("battle_of_the_sexes", "Battle of the Sexes",                     #        Ballet  Fight
                                      ["Ballet", "Fight"], ["Ballet", "Fight"],                         #Ballet  2,1     0,0
                                      [[2, 0], [0, 1]],                                                 #Fight   0,0     1,2
                                      [[1, 0], [0, 2]])

def _rock_paper_scissors_easy():
    return pyspiel.create_matrix_game("rock_paper_scissors", "Rock Paper Scissors",                     #           Rock    Paper   Scissors
                                      ["Rock", "Paper",  "Scissors"],["Rock", "Paper",  "Scissors"],    #Rock        0,0    -1,1      1,-1
                                      [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],                             #Paper       1,-1    0,0     -1,1
                                      [[0, 1, -1], [-1, 0, 1], [1, -1, 0]])                             #Scissor    -1,1     1,-1     0,0

"""Evaluates `agents1` against `agents2` for `num_episodes`."""
def eval_against_agents(env, agents1, agents2, num_episodes):
  wins = np.zeros(2)
  for player_pos in range(2):
    if player_pos == 0:
      cur_agents = [agents1[0], agents2[1]]
    else:
      cur_agents = [agents2[0], agents1[1]]
    for _ in range(num_episodes):
      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
        time_step = env.step([agent_output.action])
      if time_step.rewards[player_pos] > 0:
        wins[player_pos] += 1
  return wins / num_episodes

def train_qlearning(agents, env, training_episodes):
    for cur_episode in range(training_episodes):
        # logging every 10000 episodes
        if cur_episode % int(1e4) == 0:
            win_rates = eval_against_agents(env, agents, agents, 1000)                  #self play
            logging.info("Starting episode %s, win_rates %s", cur_episode, win_rates)

        # training
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            time_step = env.step([agent_output.action])
        # Episode is over, go over all agents with final info state.
        for agent in agents:
            agent.step(time_step)

if __name__=="__main__":
    # LOAD GAME
    game = _rock_paper_scissors_easy()
    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]
    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
        for idx in range(game.num_players())
    ]
    train_qlearning(agents, env, int(5e4))
    # Quick test: inspect top-left utility values:
    print("Values for joint action ({},{}) is {},{}".format(
        game.row_action_name(0), game.col_action_name(0),
        game.player_utility(0, 0, 0), game.player_utility(1, 0, 0)))
    # Print the initial state
    state = game.new_initial_state()
    print("State:")
    print(str(state))

    # PLAY
    assert state.is_simultaneous_node()
    chosen_actions = [
        random.choice(state.legal_actions(pid))
        for pid in range(game.num_players())
    ]
    print("Chosen actions: ", [
        state.action_to_string(pid, action)
        for pid, action in enumerate(chosen_actions)
    ])
    state.apply_actions(chosen_actions)

    # END OF GAME
    assert state.is_terminal()
    returns = state.returns()
    for pid in range(game.num_players()):
        print("Utility for player {} is {}".format(pid, returns[pid]))