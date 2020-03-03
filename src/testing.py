import logging
import numpy as np
from absl import app
from absl import flags
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent, tabular_qlearner

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", int(5e4), "Number of train episodes.")

"""Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  wins = np.zeros(2)
  for player_pos in range(2):
    if player_pos == 0:
      cur_agents = [trained_agents[0], random_agents[1]]
    else:
      cur_agents = [random_agents[0], trained_agents[1]]
    for _ in range(num_episodes):
      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
        time_step = env.step([agent_output.action])
      if time_step.rewards[player_pos] > 0:
        wins[player_pos] += 1
  return wins / num_episodes


def main(i):
    game = "tic_tac_toe"
    num_players = 2
    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]
    #Create RL_agents (Qlearning) and Random agents to test against
    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    #TRAINING THE RL_AGENTS
    training_episodes = FLAGS.num_episodes
    for cur_episode in range(training_episodes):
        # logging every 10000 episodes
        if cur_episode % int(1e4) == 0:
            win_rates = eval_against_random_bots(env, agents, random_agents, 1000)
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
    app.run(main)
