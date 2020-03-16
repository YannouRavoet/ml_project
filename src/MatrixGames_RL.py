from utils import _battle_of_the_sexes_easy
from utils import _phaseplot, _trajectoryplot
import numpy as np
import pyspiel
from absl import app
from absl import flags
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner

from algorithms import epsilongreedy_QLearner,boltzmann_QLearner, boltzmann_FAQLeaner,  boltzmann_LFAQLearner

## removing the randomness, brings the action-probabilities to a pure strategy Nash equilibrium
## intuition lenient:   learning rate can be high since we are likely to have a good action between the k action
##                      k doesn't have to be very high, since with only 2 or 3 actions for 2 players, there is a fairly high chance of having the Pareto-optimal state
##                      most interesting games to look at are probably coordination games: stag hunt.

#TODO: Dit zijn de waardes die per algoritme/per game wat aangepast moeten worden
#TODO: hou misschien een tabel bij voor elk algoritme/game zodat we dat in het report kunnen zetten als bijlage
#TODO: eps is een tricky one (die gaat altijd direct naar een uiterste, en ik vind niet waarom)
#TODO: lfaq heeft veel iteraties nodig
FLAGS = flags.FLAGS                                                                 #these values also depend on the game you test on
                                                                                    #and are thus not perfect for each game
flags.DEFINE_string("learner", "lfaq", "name of the learner")                       #options:   eps         boltz       faq         lfaq
flags.DEFINE_float("lr", 0.4, "learning rate")                                      #options:   0.001       0.01        0.1         0.5    a lower lr means slower convergence = prettier plots
flags.DEFINE_float("expl", 0.9, "initial exploration rate")                         #options:   1           0.6         1           1
flags.DEFINE_float("expl_ann", 0.999, "explorate annealing rate")                   #options:   0.99        0.99        0.999       0.999
flags.DEFINE_float("expl_min", 0.005, "minimum exploration value")                  #options:   0           0.003       0.003       0.003
flags.DEFINE_float("beta", 0.001,"(frequency adjusted) beta-value")                 #options:   /           /           0.01        0.01
flags.DEFINE_integer("k", 3, "(lenient) k-value")                                   #options:   /           /           /           8
flags.DEFINE_integer("train_iter",int(2e5),"number of training iterations")         #options:   5e2         5e2         1e4         5e5
flags.DEFINE_integer("pop_iter", 1, "number of times to train a set of agents")     #options:   7           7           10          4
#TODO: ik doe meestal pop_iter=1 totdat ik deftige waardes vind, en dan pop_iter = [7..10] naargelang wat te vol wordt op de plot

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
            env.step(action_list)
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
    state_history = []
    for cur_episode in range(training_episodes):
        #if cur_episode % int(training_episodes/3) == 0:
            #eval_against_agents(env, agents, random_agents, eval_episodes)
        agent_outputs, action_list = _env_play_episode(env, agents, evaluating=False)
        state_history.append([step_output.probs for step_output in agent_outputs])
    return state_history

def _env_play_episode(env, agents, evaluating=False):
    time_step = env.reset()
    agent_outputs = [agent.step(time_step, is_evaluation = evaluating) for agent in agents]                 #agent step (no training since no last state information) => only action selection
    action_list = [agent_output.action for agent_output in agent_outputs]                                   #reformating StepOutput to [actions]
    time_step = env.step(action_list)                                                                       #progressing the environment
    for agent in agents:
        agent.step(time_step, is_evaluation = evaluating)                                                   #preparing agents for next episode AND/OR training
    return agent_outputs, action_list

def play_episode(env, agents):
    # PLAY
    agents_output, action_list = _env_play_episode(env, agents, evaluating=True)

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
    print("-" * 80)

def main(_):
    # LOAD GAMES
    # print(pyspiel.registered_games())
    # games = [pyspiel.load_game("matrix_sh"), pyspiel.load_game("matrix_rps"), pyspiel.load_game("matrix_mp"), pyspiel.load_game("matrix_pd"),  _battle_of_the_sexes_easy()]
    games = [pyspiel.load_game("matrix_sh")] #TODO: ik doe meestal game per game (enkel de 2x2 game is in orde - trajectoryplot werkt niet voor 3x3)
    for game in games:
        # GAME INFO
        # print(game.get_type().long_name.upper())
        # state = game.new_initial_state()
        # print(state)
        # print("-"*80)

        # PLOTTING
        #_phaseplot(game)

        population_histories = []
        player1_probs = []
        player2_probs = []
        for _ in range(FLAGS.pop_iter):
            env = rl_environment.Environment(game=game)
            num_actions = env.action_spec()["num_actions"]
            agents = []
            for idx in range(env.num_players):
                if FLAGS.learner == "eps":
                    agents.append(epsilongreedy_QLearner.EpsilonGreedy_QLearner(player_id=idx, num_actions=num_actions, step_size=FLAGS.lr, discount_factor=1, epsilon=FLAGS.expl, epsilon_annealing=FLAGS.expl_ann, epsilon_min=FLAGS.expl_min))
                elif FLAGS.learner == "boltz":
                    agents.append(boltzmann_QLearner.Boltzman_QLearner(player_id=idx, num_actions=num_actions, step_size=FLAGS.lr, discount_factor=1, temperature=FLAGS.expl, temperature_annealing=FLAGS.expl_ann, temperature_min=FLAGS.expl_min))
                elif FLAGS.learner == "faq":
                    agents.append(boltzmann_FAQLeaner.Boltzmann_FAQLearner(player_id=idx, num_actions=num_actions, step_size=FLAGS.lr, discount_factor=1, temperature=FLAGS.expl, temperature_annealing=FLAGS.expl_ann, temperature_min=FLAGS.expl_min, beta=FLAGS.beta))
                else:
                    agents.append(boltzmann_LFAQLearner.Boltzmann_LFAQLearner(player_id=idx, num_actions=num_actions, step_size=FLAGS.lr,discount_factor=1, temperature=FLAGS.expl, temperature_annealing=FLAGS.expl_ann,temperature_min=FLAGS.expl_min, beta=FLAGS.beta, k=FLAGS.k))

            random_agents = [
                random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
                for idx in range(env.num_players)
            ]
            # PLAY BEFORE TRAIN
            # print("BEFORE TRAINING: 1 episode of self-play")
            # play_episode(env, agents)

            # TRAIN
            history = train_qlearning(agents, env, FLAGS.train_iter, random_agents)       #needs to be high for LFAQ
            population_histories.append(history)
            agents_output, _ = _env_play_episode(env, agents, evaluating=True)
            player1_probs.append(agents_output[0].probs)
            player2_probs.append(agents_output[1].probs)

            # PLAY AFTER TRAIN
            # print("AFTER TRAINING: 1 episode of self-play")
            # play_episode(env, agents)
            # print("-"*80)
        #TODO: na een mooie plot kunt ge die saven onder ml_project/resources/plots/trajectory/<algoritme>/<game>.png
        _trajectoryplot(game, population_histories)

        for i in range(len(player1_probs)):
            print(f"\t\tPlayer 1\t Player 2")
            print(f"{env.get_state.action_to_string(0, 0)}:\t{player1_probs[i][0]:.2f}\t\t{player2_probs[i][0]:.2f}")
            print(f"{env.get_state.action_to_string(0, 1)}:\t{player1_probs[i][1]:.2f}\t\t{player2_probs[i][1]:.2f}")
            print()



if __name__=="__main__":
    app.run(main)