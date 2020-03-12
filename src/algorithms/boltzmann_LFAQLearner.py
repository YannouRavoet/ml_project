from boltzmann_FAQLeaner import Boltzmann_FAQLearner
from collections import defaultdict
from open_spiel.python import rl_agent
import numpy as np

class Boltzmann_LFAQLearner(Boltzmann_FAQLearner):
    def __init__(self,
                 player_id,
                 num_actions,
                 step_size=0.5,
                 temperature=1,
                 temperature_annealing=0.999,
                 temperature_min=0.001,
                 discount_factor=1.0,
                 beta=0.5,
                 k = 5):
        super(Boltzmann_LFAQLearner, self).__init__(player_id, num_actions, step_size, discount_factor, temperature, temperature_annealing, temperature_min,  beta)
        self._k = k
        self._k_actions = []
        self._k_rewards = []
        self._k_probs = []

    def step(self, time_step, is_evaluation=False):
        action, info_state, legal_actions, probs = self._step_action_selection(is_evaluation, time_step)

        # Learn step: don't learn during evaluation or at first agent steps.
        if self._prev_info_state and not is_evaluation:
            self._k_rewards.append(time_step.rewards[self._player_id])
            self._k_actions.append(self._prev_action)
            self._k_probs.append(self._prev_probs)
            if len(self._k_actions) == self._k:
                self._exploration = max(self._exploration * self._exploration_annealing, self._exploration_min)       #EXPLORATION ANNEALING
                best_index = np.where(self._k_rewards == np.amax(self._k_rewards))[0][0]
                self._prev_action = self._k_actions[best_index]
                self._prev_probs = self._k_probs[best_index]
                target = self._k_rewards[best_index]
                if not time_step.last(): # no legal actions in last timestep
                    target += self._discount_factor * max(
                        [self._q_values[info_state][a] for a in legal_actions])

                prev_q_value = self._q_values[self._prev_info_state][self._prev_action]
                self._last_loss_value = target - prev_q_value                                                         #last_loss_value = target - Q(t)[prev_action]
                self._train_update()

                #reset all k_arrays
                self._k_actions = []
                self._k_rewards = []
                self._k_probs = []

            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                return

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action
            self._prev_probs = probs
        return rl_agent.StepOutput(action=action, probs=probs)
