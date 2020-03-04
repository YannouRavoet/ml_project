from utils import _prisonners_dilemma_easy, _matching_pennies_easy, _battle_of_the_sexes_easy, _rock_paper_scissors_easy
from utils import _streamplot
import numpy as np
import pyspiel
from absl import app
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner


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
            res = time_step.is_simultaneous_move()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                time_step = env.step([agent_output.action])
            if time_step.rewards[player_pos] > 0:
                wins[player_pos] += 1
    return wins / num_episodes


def train_qlearning(agents, env, training_episodes):
    for cur_episode in range(training_episodes):
        # if cur_episode % int(1e4) == 0:
        #     win_rates = eval_against_agents(env, agents, agents, 1000)
        #     logging.info("Starting episode %s, win_rates %s", cur_episode, win_rates)
        time_step = env.reset()
        while not time_step.last():
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
    # print(pyspiel.registered_games())
    # games = [_rock_paper_scissors_easy(), _battle_of_the_sexes_easy(), _matching_pennies_easy(),
    #          _prisonners_dilemma_easy()]
    games = [pyspiel.load_game("matrix_rps"), pyspiel.load_game("matrix_mp"), pyspiel.load_game("matrix_pd"),  _battle_of_the_sexes_easy()]
    for game in games:
        print(game.get_type().long_name.upper())
        state = game.new_initial_state()
        print(state)
        print("-"*80)

        #PLOTTING
        _streamplot(game)
        # RL Part
        env = rl_environment.Environment(game=game)
        env.reset()
        num_actions = env.action_spec()["num_actions"]
        agents = [
            tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
            for idx in range(env.num_players)
        ]
        # PLAY BEFORE TRAIN
        print("BEFORE TRAINING")
        play_episode(env, agents)

        # TRAIN
        train_qlearning(agents, env, int(1e4))

        # PLAY AFTER TRAIN
        print("AFTER TRAINING")
        play_episode(env, agents)
        print("-"*80)


if __name__=="__main__":
    app.run(main)