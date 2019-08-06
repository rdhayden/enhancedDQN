import pickle
import torch
import collections
from lib.network import Network
from lib.experience_buffer import ExperienceBuffer
from lib.extended_game import ExtendedGame
from lib.agent import Agent
from lib.utils import calc_loss
from tqdm import trange
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import datetime
import os, sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plays ViZDoom agent with given model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/deathAmmoPenalty.pth",
        help="Boolean to make game window visible. Default: False",
    )
    parser.add_argument(
        "--gpu", type=bool, default=True, help="Boolean to use GPU. Default: True"
    )
    parser.add_argument(
        "--num_frames_stacked",
        type=int,
        default=4,
        help="Number of frames to stack together. Default: 4",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=54,
        help="Width to resize image to: Default 54",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=54,
        help="Height to resize image to. Default: 54",
    )
    parser.add_argument(
        "--game_config_file_path",
        type=str,
        default="scenarios/defend_the_line.cfg",
        help="Path to game config file. Default: scenarios/defend_the_line.cfg",
    )
    parser.add_argument(
        "--repeat_for_frames",
        type=int,
        default=4,
        help="Number of frames to repeat each action. Default: 4",
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of epochs. Default: 500"
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=2000,
        help="Number of training steps per epoch. Default: 2000",
    )
    parser.add_argument(
        "--health_reward",
        type=int,
        default=0,
        help="sets the reward for negative change in health. Default: 0",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if args.gpu else "cpu")

    # Initialise the game instance
    game = ExtendedGame(args.game_config_file_path)
    game.set_window_visible(True)
    game.init()

    # Setup and load the network
    net_shape = (args.num_frames_stacked, args.width, args.height)
    net = Network(net_shape, game.n_actions).to(device)
    net.load_state_dict(torch.load(args.model))
    net.eval()

    # Initialise the agent
    agent = Agent(
        game,
        net,
        None,
        args.width,
        args.height,
        args.num_frames_stacked,
        args.repeat_for_frames,
        args.health_reward,
    )

    done_reward = agent.play_step(False, device, write_to_buffer=False)
    while(not done_reward):
        done_reward = agent.play_step(False, device, write_to_buffer=False)
    
    print(done_reward)