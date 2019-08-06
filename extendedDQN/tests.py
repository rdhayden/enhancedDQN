import torch
import torch.optim as optim
import collections
from lib.network import Network
from lib.experience_buffer import ExperienceBuffer
from lib.extended_game import ExtendedGame
from lib.agent import Agent
from lib.utils import calc_loss
from tqdm import trange
import numpy as np
import torch.nn as nn
from skimage import io
from matplotlib import pyplot as plt

if __name__ == "__main__":

    print("After running these tests it is advisable to run the following to ensure the whole thing runs together:")
    print()
    print("python vd_learning.py --troubleshoot True")
    print()
    print("Also note that where a single image of a frame is shown, you will need to close it to continue")
    print()
    input("Press any key to continue")

    print("Test ExtendedGame: sample_actions returns int between 1 and 3")
    device = torch.device("cuda")
    game = ExtendedGame("scenarios/basic.cfg")
    for i in range(10):
        action = game.sample_actions()
        assert type(action) is int, "fail: action is not an int"
        assert 0 <= action < 5, "fail: action is not between 1 and 3"
    print("pass: action is an int and is between 1 and 3")

    print(
        "Visually test ExtendedGame: start game, move right 4 times, move left 4 times, turn left 4 times, turn right 4 times, fire 12 times"
    )
    device = torch.device("cuda")
    game = ExtendedGame("scenarios/defend_the_line.cfg")
    game.set_window_visible(True)
    game.init()

    input("Press any key to make move left action 4 times, each action repeats for 4 frames")
    reward = 0
    for i in range(4):
        reward += game.make_action_int(0, 4)
    print("reward {}".format(reward))

    input(
        "Press any key to make move right action 4 times, each action repeats for 4 frames"
    )
    reward = 0
    for i in range(4):
        reward += game.make_action_int(1, 4)
    print("reward {}".format(reward))

    input(
        "Press any key to make turn left action 4 times, each action repeats for 4 frames"
    )
    reward = 0
    for i in range(4):
        reward += game.make_action_int(2, 4)
    print("reward {}".format(reward))

    input(
        "Press any key to make turn right action 4 times, each action repeats for 4 frames"
    )
    reward = 0
    for i in range(4):
        reward += game.make_action_int(3, 4)
    print("reward {}".format(reward))

    input(
        "Press any key to make fire action 12 times, each action repeats for 4 frames"
    )
    reward = 0
    for i in range(12):
        reward += game.make_action_int(4, 4)
    print("reward {}".format(reward))

    input("Press any key to continue")

    capacity = 10
    nsteps = 2
    gamma = 0.99
    experience_buffer = ExperienceBuffer(capacity, nsteps, gamma)
    num_frames_stacked = 4
    width = 54
    height = 54
    repeat_for_frames = 4
    net_shape = (num_frames_stacked, width, height)
    train_net = Network(net_shape, game.n_actions).to(device)
    target_net = Network(net_shape, game.n_actions).to(device)
    agent = Agent(
        game,
        train_net,
        experience_buffer,
        width,
        height,
        num_frames_stacked,
        repeat_for_frames,
    )

    print(
        "Test Agent: state should be shaped num frames {} width {} height {}".format(
            num_frames_stacked, width, height
        )
    )
    assert agent.state.shape == (
        num_frames_stacked,
        width,
        height,
    ), "fail: agent state should be shaped {} {} {}".format(
        num_frames_stacked, width, height
    )
    print("pass: agent state shaped {} {} {}".format(num_frames_stacked, width, height))

    print(
        "Test Agent: initially all frames in the stack except the top frame should be zero"
    )
    assert (
        sum(agent.state[0 : num_frames_stacked - 1].flatten()) == 0
    ), "fail: first 3 frames in the stack should be zero when initialised"
    print("pass: first 3 frames in the stack are zero when initialised")
    assert (
        sum(agent.state[num_frames_stacked - 1].flatten()) != 0
    ), "fail: the last frame in the stack should not be zero when initialised"
    print("pass: the last frame in the stack is not zero when initialised")

    print(
        "Test Agent: reset after stack is full should result in all frames in the stack except the top frame being zero"
    )
    agent.state = np.random.rand(num_frames_stacked, width, height)
    agent._reset()
    assert (
        sum(agent.state[0 : num_frames_stacked - 1].flatten()) == 0
    ), "fail: first 3 frames in the stack should be zero"
    print("pass: first 3 frames in the stack are zero")
    assert (
        sum(agent.state[num_frames_stacked - 1].flatten()) != 0
    ), "fail: the last frame in the stack should not be zero"
    print("pass: the last frame in the stack should not be zero")

    print(
        "Visualy test Agent, after reset top frame should be a reduced image of size {} by {}".format(
            width, height
        )
    )
    agent._reset()
    io.imshow(agent.state[-1])
    plt.show()
    input("Press any key to continue")
    plt.close()

    print(
        "resetting experience buffer, then observe state and invoke agent play method 100 times ..."
    )
    experience_buffer.buffer.clear()
    agent._reset()
    for i in range(100):
        play_random = True
        agent.play_step(play_random)
    print(
        "number of experiences in the buffer: {}. Buffer capacity {}".format(
            len(experience_buffer), capacity
        )
    )

    input(
        "Press any key to see a random initial state image from the experience buffer"
    )
    states, actions, rewards, dones, next_states = experience_buffer.sample(1)
    io.imshow(states[0][3])
    plt.show()
    input("Press any key to see the next state image for the same experience")
    plt.close()
    io.imshow(next_states[0][3])
    plt.show()
    input("Press any key to continue")

    print(
        "Test Agent: after some iterations check if the second frame in the stack is not zero"
    )
    agent._reset()
    second_frame_non_zero = False
    for i in range(100):
        play_random = True
        agent.play_step(play_random)
        if sum(agent.state[1].flatten()) != 0:
            second_frame_non_zero = True
            break
    assert second_frame_non_zero, "fail: 2nd frame never non-zero after 100 steps"
    print("pass: 2nd frame non-zero at least some of the time")

    print("Network Agent: test network outputs with framestack input")
    state_a = np.array([agent.state], copy=False)
    state_tensor = torch.tensor(state_a, dtype=torch.float).to(device)
    q_vals_tensor = train_net(state_tensor)
    print("Q value predictions:")
    print(q_vals_tensor)
    print()
    _, index = torch.max(
        q_vals_tensor, 1
    )  # index of action predicted with the highest value
    action = index.cpu().data.numpy()[0]
    print("Predicted action: {}".format(action))
