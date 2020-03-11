from boltzmann_QLearner import Boltzman_QLearner
from open_spiel.python import rl_agent

class Boltzmann_FAQLearner(Boltzman_QLearner):
    def __init__(self,
                 player_id,
                 num_actions,
                 step_size=0.5,
                 discount_factor=1.0,
                 temperature = 1,
                 temperature_annealing = 0.999,
                 temperature_min = 0.001,
                 beta = 0.5):
        super(Boltzmann_FAQLearner, self).__init__(player_id, num_actions, step_size, discount_factor, temperature, temperature_annealing, temperature_min)
        self._prev_prob = None
        self._beta = beta

    #THE UPDATE STEP IS ADJUSTED FOR THE FREQUENCY WITH WHICH AN ACTION IS SELECTED
    #AN ACTION THAT IS SELECTED INFREQUENTLY WILL HAVE BIGGER UPDATES
    def step(self, time_step, is_evaluation=False):
        """Returns the action to be taken and updates the Q-values if needed.

        Args:
          time_step: an instance of rl_environment.TimeStep.
          is_evaluation: bool, whether this is a training or evaluation call.

        Returns:
          A `rl_agent.StepOutput` containing the action probs and chosen action.
        """
        info_state = str(time_step.observations["info_state"][self._player_id])
        legal_actions = time_step.observations["legal_actions"][self._player_id]

        # Prevent undefined errors if this agent never plays until terminal step
        action, probs = None, None

        # Act step: don't act at terminal states.
        if not time_step.last():
            action, probs = self._boltzman_exploration(info_state, legal_actions)

        # Learn step: don't learn during evaluation or at first agent steps.
        if self._prev_info_state and not is_evaluation:
            #TEMPERATURE ANNEALING = EACH TIME WE TRAIN WE REDUCE THE EXPLORATION RATE
            self._temperature = max(self._temperature * self._temperature_annealing, self._temperature_min)
            # target = REWARD + DISCOUNT * MAX Q_VALUE[LEGAL ACTIONS] = r(t+1) + gamma*max_j Q_j(t)
            target = time_step.rewards[self._player_id]
            if not time_step.last():  # Q values are zero for terminal.
                target += self._discount_factor * max(
                    [self._q_values[info_state][a] for a in legal_actions])

            prev_q_value = self._q_values[self._prev_info_state][self._prev_action]
            # last_loss_value = target - Q(t)[prev_action] = r(t+1) + gamma*max_j Q_j(t) - Q_i(t)
            self._last_loss_value = target - prev_q_value
            # ONLY THE Q-value of the previous action is update based on the current and future reward
            # Q_i(t+1) = Q_i(t) + alpha * [ r(t+1) + gamma*max_j Q_j(t) - Q_i(t) ]
            #FAQ: Q_i(t+1) = Q_i(t) + min(B/x_i, 1) * alpha * [ r(t+1) + gamma*max_j Q_j(t) - Q_i(t) ]
            self._q_values[self._prev_info_state][self._prev_action] += (
                min(self._beta/self._prev_prob, 1) * self._step_size * self._last_loss_value)

            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                return

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action
            self._prev_prob = probs[action]
        return rl_agent.StepOutput(action=action, probs=probs)