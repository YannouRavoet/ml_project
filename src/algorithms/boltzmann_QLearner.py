from open_spiel.python.algorithms.tabular_qlearner import QLearner
import numpy as np

class Boltzman_QLearner(QLearner):

    def __init__(self,
                 player_id,
                 num_actions,
                 step_size=0.5,
                 discount_factor=1.0,
                 temperature = 1,
                 temperature_annealing = 0.999,
                 temperature_min = 0.005):
        super(Boltzman_QLearner, self).__init__(player_id, num_actions, step_size, epsilon=0, discount_factor=discount_factor)
        self._temperature = temperature
        self._temperature_annealing = temperature_annealing
        self._temperature_min = temperature_min

    def _init_q_values(self, info_state):
        for action in range(self._num_actions):
            self._q_values[info_state][action] = 0

    # WHEN THE TEMPERATURE IS VERY LOW probs[action] can overflow: for example: e^(2/0.0028)=undefined. Set self._temperature_min >= 0.005 to be safe
    def _boltzman_exploration(self, info_state, legal_actions):
        #init q_values if necessary
        if len(self._q_values[info_state])==0:
            self._init_q_values(info_state)
        #set the probability of non-legal actions at 0
        probs = np.zeros(self._num_actions)
        #Softmax the probability of legal actions
        for action in legal_actions:
            probs[action] =  np.exp(self._q_values[info_state][action]/self._temperature) \
                             /  sum( np.exp(np.array(list(self._q_values[info_state].values()))/self._temperature) )
        #Choose an action based on the calculated probabilities
        action = np.random.choice(range(self._num_actions), p=probs)
        return action, probs


    def _epsilon_greedy(self, info_state, legal_actions, epsilon):
        return self._boltzman_exploration(info_state, legal_actions)

    #TRAINING IS INDEPENDENT OF THE POLICY (epsilon-greedy vs Boltzmann)
    #=> Policy is only used to select the next action
    def step(self, time_step, is_evaluation=False):
        #IF YOU TRAIN: REDUCE TEMPERATURE
        if self._prev_info_state and not is_evaluation:
            self._temperature = max(self._temperature * self._temperature_annealing, self._temperature_min)
        return super(Boltzman_QLearner, self).step(time_step, is_evaluation)