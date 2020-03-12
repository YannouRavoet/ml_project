from boltzmann_QLearner import Boltzman_QLearner

class Boltzmann_FAQLearner(Boltzman_QLearner):
    def __init__(self,
                 player_id,
                 num_actions,
                 step_size=0.5,
                 temperature = 1,
                 temperature_annealing = 0.999,
                 temperature_min = 0.001,
                 discount_factor=1.0,
                 beta = 0.5):
        super(Boltzmann_FAQLearner, self).__init__(player_id, num_actions, step_size, temperature, temperature_annealing, temperature_min, discount_factor)
        self._beta = beta
        self._prev_probs = None

    def step(self, time_step, is_evaluation=False):
        step_output = super(Boltzmann_FAQLearner, self).step(time_step, is_evaluation)
        if not time_step.last() and not is_evaluation:
            self._prev_probs = step_output.probs
        return step_output

    def _train_update(self):
        # FAQ: Q_i(t+1) = Q_i(t) + min(B/x_i, 1) * alpha * [ r(t+1) + gamma*max_j Q_j(t) - Q_i(t) ]
        self._q_values[self._prev_info_state][self._prev_action] += 
            (min(self._beta / self._prev_probs[self._prev_action], 1) * self._step_size * self._last_loss_value)

            
            
