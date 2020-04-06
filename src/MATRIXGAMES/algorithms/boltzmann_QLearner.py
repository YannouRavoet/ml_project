from tabular_QLearner import QLearner
import numpy as np

class Boltzman_QLearner(QLearner):

    def __init__(self,
                 player_id,
                 num_actions,
                 step_size=0.5,
                 temperature = 1,
                 temperature_annealing = 0.999,
                 temperature_min = 0.005,
                 discount_factor=1.0):
        super(Boltzman_QLearner, self).__init__(player_id, num_actions, step_size, temperature, temperature_annealing, temperature_min, discount_factor)

    # WARNING: when the temperature is very low, probs[action] can overflow. for example: e^(2/0.0028)=undefined.
    # Set self._exploration_min >= 0.005 to be safe
    def _exploration_function(self, info_state, legal_actions, temperature):
        #set the probability of non-legal actions at 0
        probs = np.zeros(self._num_actions)
        #softmax the probability of legal actions
        for action in legal_actions:
            probs[action] =  np.exp(self._q_values[info_state][action]/temperature) \
                             /  sum( np.exp(np.array(list(self._q_values[info_state].values()))/temperature) )
        #Choose an action based on the calculated probabilities
        action = np.random.choice(range(self._num_actions), p=probs)
        return action, probs


