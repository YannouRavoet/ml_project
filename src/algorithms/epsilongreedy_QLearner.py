from tabular_QLearner import QLearner
import numpy as np

class EpsilonGreedy_QLearner(QLearner):
    def __init__(self,
                 player_id,
                 num_actions,
                 step_size=0.5,
                 epsilon=0.2,
                 epsilon_annealing=0.999,
                 epsilon_min=0.001,
                 discount_factor=1.0):
        super(EpsilonGreedy_QLearner, self).__init__(player_id, num_actions, step_size, epsilon, epsilon_annealing, epsilon_min, discount_factor)

    #source: open_spiel.python.algorithms.tabular_qlearner
    def _exploration_function(self, info_state, legal_actions, epsilon):
        """Returns a valid epsilon-greedy action and valid action probs.

            If the agent has not been to `info_state`, a valid random action is chosen.

            Args:
              info_state: hashable representation of the information state.
              legal_actions: list of actions at `info_state`.
              epsilon: float, prob of taking an exploratory action.

            Returns:
              A valid epsilon-greedy action and valid action probabilities.
            """
        probs = np.zeros(self._num_actions)
        greedy_q = max([self._q_values[info_state][a] for a in legal_actions])
        greedy_actions = [
            a for a in legal_actions if self._q_values[info_state][a] == greedy_q
        ]
        probs[legal_actions] = epsilon / len(legal_actions)
        probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)
        action = np.random.choice(range(self._num_actions), p=probs)
        return action, probs