import random
import numpy as np
import pyspiel
from open_spiel.python.egt import alpharank

game = pyspiel.load_game("connect_four")
state = game.new_initial_state()
while not state.is_terminal():
    legal_actions = state.legal_actions()
    action = legal_actions[0]
    state.apply_action(action)
