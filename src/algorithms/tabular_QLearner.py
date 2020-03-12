"""Abstract Tabular Q-learning agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import abc
from open_spiel.python import rl_agent
import random

#based on open_spiel.python.algorithms.tabular_qlearner
#but has a flexible exploration function
#random q_value initialization (for population trajectory plots = different starting points)
class QLearner(rl_agent.AbstractAgent):
    def __init__(self,
                 player_id,
                 num_actions,
                 step_size,
                 exploration,
                 exploration_annealing,
                 exploration_min,
                 discount_factor):
        """Initialize the Q-Learning agent."""
        self._player_id = player_id
        self._num_actions = num_actions
        self._step_size = step_size
        self._exploration = exploration
        self._exploration_annealing = exploration_annealing
        self._exploration_min = exploration_min
        self._discount_factor = discount_factor
        self._q_values = collections.defaultdict(
            lambda: collections.defaultdict(float))
        self._prev_info_state = None
        self._prev_action = None
        self._last_loss_value = 0
        self._init_q_values('[0.0]')

    def _init_q_values(self, info_state):
        for action in range(self._num_actions):
            self._q_values[info_state][action] = random.random()

    @abc.abstractmethod
    def _exploration_function(self, info_state, legal_actions, exploration):
        """Defines the exploration function (epsilon-greedy, boltzmann)

        Args:
          info_state: hashable representation of the information state.
          legal_actions: list of actions at `info_state`.
          exploration: float, exploration vs exploitation term.

        Returns:
          A valid action and valid action probabilities.
          """

    def step(self, time_step, is_evaluation=False):
        """Returns the action to be taken and updates the Q-values if needed.

            Args:
            time_step: an instance of rl_environment.TimeStep.
            is_evaluation: bool, whether this is a training or evaluation call.

            Returns:
            A `rl_agent.StepOutput` containing the action probs and chosen action.
        """
        action, info_state, legal_actions, probs = self._step_action_selection(is_evaluation, time_step)

        # Learn step: don't learn during evaluation or at first agent steps.
        if self._prev_info_state and not is_evaluation:
            self._exploration = max(self._exploration * self._exploration_annealing, self._exploration_min)       #EXPLORATION ANNEALING
            target = time_step.rewards[self._player_id]                                                           #target = REWARD + DISCOUNT * MAX Q_VALUE[LEGAL ACTIONS]
            #IN A ONE-SHOT GAME FUTURE REWARDS WILL NEVER BE USED
            if not time_step.last(): # Q values are zero for terminal.
                target += self._discount_factor * max(
                    [self._q_values[info_state][a] for a in self._num_actions])

            prev_q_value = self._q_values[self._prev_info_state][self._prev_action]
            self._last_loss_value = target - prev_q_value                                                         #last_loss_value = target - Q(t)[prev_action]
            self._train_update()

            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                return

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action
        return rl_agent.StepOutput(action=action, probs=probs)

    def _step_action_selection(self, is_evaluation, time_step):
        info_state = str(time_step.observations["info_state"][self._player_id])
        legal_actions = time_step.observations["legal_actions"][self._player_id]
        # Prevent undefined errors if this agent never plays until terminal step
        action, probs = None, None
        # Act step: don't act at terminal states.
        if not time_step.last():
            # the reason this is not ..= 0 if ... is that this divides by zero for the boltzmann implementation.
            exploration = self._exploration_min if is_evaluation else self._exploration
            action, probs = self._exploration_function(info_state, legal_actions, exploration)
        return action, info_state, legal_actions, probs

    def _train_update(self):
        # ony the Q-value of the previous action is updated based on the current and future reward
        # Q_i(t+1) = Q_i(t) + alpha * [ r(t+1) + gamma*max_j Q_j(t) - Q_i(t) ]
        self._q_values[self._prev_info_state][self._prev_action] += (self._step_size * self._last_loss_value)


@property
def loss(self):
    return self._last_loss_value
