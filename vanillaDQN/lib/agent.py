from lib.experience_buffer import Experience
from skimage.transform import resize
import numpy as np
import torch
import time

class Agent:
    def __init__(
        self, 
        game, 
        net, 
        exp_buffer, 
        width, 
        height, 
        num_frames_stacked, 
        repeat_for_frames, 
        record=False
    ):
        self.game = game
        self.net = net
        self.exp_buffer = exp_buffer
        self.width = width
        self.height = height
        self.num_frames_stacked = num_frames_stacked
        self.repeat_for_frames = repeat_for_frames

        if record:
            self.record_path = str(time.time()) + '.lmp'
        else:
            self.record_path = ""
        
        self._reset()

    def _reset(self):
        self.game.new_episode(self.record_path)
        # reset the stack of frames to zero, we'll append a frame at each step of the game
        self.state = np.zeros((self.num_frames_stacked, self.width, self.height))
        self.state = self._process_frame(self.game.get_state().screen_buffer)
    
    # Down-samples the image and stacks the frame
    def _process_frame(self, img):
        # remove the cieling, it doesn't add any information, and reduce width 
        # both sides to create square image as the view is always centered on the player
        img = img[96:, 88:232]

        # resize the image
        resized_img = resize(img, (self.width, self.height))
        
        # each frame needs another dimension so it can be stacked
        new_frame = np.array([resized_img])

        # It appears resize converts to a float between 0 and 1 so no need for the next conversion 
        # new_frame = new_frame.astype(np.float32) / 255.0

        # now create stack with last few frames and this frame
        new_stack = np.copy(self.state)
        new_stack[:-1] = new_stack[1:]
        new_stack[-1] = new_frame
        return new_stack

    def play_step(self, play_random, device="cpu", write_to_buffer=True):
        done_reward = None
        if play_random:
            action = self.game.sample_actions()
        else:
            # add one dimension the array
            state_a = np.array([self.state], copy=False)
            state_tensor = torch.tensor(state_a, dtype=torch.float).to(device)
            q_vals_tensor = self.net(state_tensor)
            _, index = torch.max(
                q_vals_tensor, 1
            ) # index of action predicted with the highest value
            action = index.cpu().data.numpy()[0]

        # take actions in the game
        reward = self.game.make_action_int(action, self.repeat_for_frames)
        is_done = self.game.is_episode_finished()

        if is_done:
            done_reward = self.game.get_total_reward()
            new_state = (
                self.state
             ) # it will get overriden later anyway so this is just a placeholder
            exp = Experience(self.state, action, reward, is_done, new_state)
            if write_to_buffer:
                self.exp_buffer.append(exp)
            self._reset()
        else:
            new_state = self._process_frame(self.game.get_state().screen_buffer)
            exp = Experience(self.state, action, reward, is_done, new_state)
            if write_to_buffer:
                self.exp_buffer.append(exp)
            self.state = new_state

        return done_reward
