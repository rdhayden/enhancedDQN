from vizdoom import DoomGame
from vizdoom import Button
import numpy as np
import itertools as it


class ExtendedGame(DoomGame):
    # Extend the Doom game to make it easy to make actions with integer references 
    # and to make it easy to sample random actions
    def __init__(self, config_file_path):
        super(ExtendedGame, self).__init__()
        self.load_config(config_file_path)

        # MOVE_LEFT [1,0,0,0,0]
        # MOVE_RIGHT [0,1,0,0,0]
        # TURN_LEFT [0,0,1,0,0]
        # TURN_RIGHT [0,0,0,1,0]
        # ATTACK [0,0,0,0,1]
        self.actions = [
            [1, 0, 0, 0, 0], 
            [0, 1, 0, 0, 0], 
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]

        self.n_actions = len(self.actions)

    def sample_actions(self):
        return np.random.choice(self.n_actions)

    def make_action_int(self, action_int, repeat_for_frames):
        '''a utility method to make it possible to make actions with integers'''
        return self.make_action(self.actions[action_int], repeat_for_frames)
