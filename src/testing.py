import random
import numpy as np
import pyspiel


game = pyspiel.load_game("connect_four")
state = game.new_initial_state()
while not state.is_terminal():
    legal_actions = state.legal_actions()
