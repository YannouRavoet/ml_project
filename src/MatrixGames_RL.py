from utils import _prisonners_dilemma_easy, _matching_pennies_easy, _battle_of_the_sexes_easy, _rock_paper_scissors_easy
from utils import _phaseplot, _trajectoryplot
import numpy as np
import pyspiel
import logging
from absl import app
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.algorithms import random_agent

#source: open_spiel/python/examples/tic_tac_toe_qlearner.py
"""Evaluates `trained_agents` against `eval_agents` for `num_episodes`."""
def eval_against_agents(env, trained_agents, eval_agents, num_episodes):
    # WARNING: only valid for 2 players!!!                      #               Player 1   Player2
    returns_agents = np.zeros((2,2))                            #trained agents     0       0
                                                                #eval agents        0       0
    for batch in range(2):
        #FIRST BATCH:   Trained Player 1 vs Eval Player 2
        #SECOND BATCH:  Eval Player 1 vs Trained Player 2
        cur_agents = [trained_agents[0], eval_agents[1]] if batch==0 else [eval_agents[0], trained_agents[1]]
        for _ in range(num_episodes):
            time_step = env.reset()
            #while not time_step.last(): #not really necessary for ONE-SHOT GAME...
            agents_output = [agent.step(time_step, is_evaluation=True) for agent in cur_agents]
            action_list = [agent_output.action for agent_output in agents_output]
            time_step = env.step(action_list)
            if batch==0:
                returns_agents[0][0] += env.get_state.returns()[0]          #TRAINED AGENT RETURN
                returns_agents[1][1] += env.get_state.returns()[1]          #EVAL AGENT RETURN
            else:
                returns_agents[0][1] += env.get_state.returns()[1]          #TRAINED AGENT RETURN
                returns_agents[1][0] += env.get_state.returns()[0]          #EVAL AGENT RETURN

    returns_agents /= num_episodes
    print(f"Average return over {num_episodes} episodes:")
    print(f"Trained agents: Player 1={returns_agents[0][0]} - Player 2={returns_agents[0][1]}")
    print(f"Eval agents: Player 1={returns_agents[1][0]} - Player 2={returns_agents[1][1]}")


def train_qlearning(agents, env, training_episodes, random_agents):
    eval_episodes = 1000
    for cur_episode in range(training_episodes):
        if cur_episode % int(1e4) == 0:
            eval_against_agents(env, agents, random_agents, eval_episodes)
        time_step = env.reset()
        while not time_step.last(): #NOT REALLY NECESSARY FOR ONE-SHOT GAMES
            agents_output = [agent.step(time_step, is_evaluation=False) for agent in agents]    #agent step and training
            action_list = [agent_output.action for agent_output in agents_output]               #reformating StepOutput to [actions]
            time_step = env.step(action_list)                                                   #progressing the environment

        for agent in agents:
            agent.step(time_step)                                                               #resetting the agents

def play_episode(env, agents):
    # PLAY
    time_step = env.reset()
    agents_output = [agent.step(time_step) for agent in agents]
    action_list = [agent_output.action for agent_output in agents_output]
    time_step = env.step(action_list)
    for agent in agents:
        agent.step(time_step)

    # PRINT
    for pid, step_output in enumerate(agents_output):
        print(f"Action probabilities player {pid}: ")
        for action, prob in enumerate(step_output.probs):
            print(f"{env.get_state.action_to_string(pid, action)} : {prob}")

    print("Chosen actions: ", [
        env.get_state.action_to_string(pid, action)
        for pid, action in enumerate(action_list)
    ])
    returns = env.get_state.returns()
    for pid in range(env.num_players):
        print("Utility for player {} is {}".format(pid, returns[pid]))
    print("-"*80)


def main(_):
    # LOAD GAMES
    print(pyspiel.registered_games())
    # games = [_rock_paper_scissors_easy(), _battle_of_the_sexes_easy(), _matching_pennies_easy(),
    #          _prisonners_dilemma_easy()]
    games = [pyspiel.load_game("matrix_sh"), pyspiel.load_game("matrix_mp"), pyspiel.load_game("matrix_pd"),  _battle_of_the_sexes_easy(), pyspiel.load_game("matrix_rps")]
    for game in games:
        print(game.get_type().long_name.upper())
        state = game.new_initial_state()
        print(state)
        print("-"*80)

        #PLOTTING
        _phaseplot(game)
        # RL Part
        env = rl_environment.Environment(game=game)
        env.reset()
        num_actions = env.action_spec()["num_actions"]
        agents = [
            #removing the randomness, brings the action-probabilities to a Nash equilibrium.
            tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions, step_size=0.5, epsilon=0)
            for idx in range(env.num_players)
        ]
        random_agents = [
            random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
            for idx in range(env.num_players)
        ]
        #RANDOM AGENTS AGAINST RANDOM AGENTS
        print(f"Baseline evaluation (untrained agents against random agents)...")
        eval_against_agents(env, agents, random_agents, 1000)
        # PLAY BEFORE TRAIN
        print("BEFORE TRAINING: 1 episode of self-play")
        play_episode(env, agents)

        # TRAIN
        train_qlearning(agents, env, int(2e4+1), random_agents)

        # PLAY AFTER TRAIN
        print("AFTER TRAINING: 1 episode of self-play")
        play_episode(env, agents)
        print("-"*80)


if __name__=="__main__":
    app.run(main)