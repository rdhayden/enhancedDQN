import pickle
import torch
import torch.optim as optim
import collections
from lib.conv_net import ConvNet
from lib.experience_buffer import ExperienceBuffer
from lib.extended_game import ExtendedGame
from lib.agent import Agent
from lib.utils import calc_loss
from tqdm import trange
import numpy as np
from tensorboardX import SummaryWriter
import time
import argparse
import datetime
import os, sys

if __name__ == "__main__":

    start_date_time = datetime.datetime.now()

    parser = argparse.ArgumentParser(description='Trains ViZDoom vanilla DQN agent')
    parser.add_argument(
        "--window_visible", 
        type=bool, default=False, 
        help="Boolean to make game window visible. Default: False"
    )
    parser.add_argument(
        "--gpu", 
        type=bool, 
        default=True, 
        help="Boolean to use GPU. Default: True"
    )
    parser.add_argument(
        "--num_frames_stacked", 
        type=int, default=4, 
        help="Number of frames to stack together. Default: 4"
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
        default="scenarios/defend_the_line_no_ammo_penalty.cfg", 
        help="Path to game config file. Default: scenarios/defend_the_line_no_ammo_penalty.cfg"
    )
    parser.add_argument(
        "--repeat_for_frames", 
        type=int, 
        default=4, 
        help="Number of frames to repeat each action. Default: 4"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=0.00001, 
        help="Learning rate for the optimizer. Default: "
    )
    parser.add_argument(
        "--sync_networks_after", 
        type=int, 
        default=500, 
        help="Steps between synchronization of target and training networks. Default: 1024"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=200, 
        help="Number of epochs. Default: 500"
    )
    parser.add_argument(
        "--steps_per_epoch", 
        type=int, 
        default=2000, 
        help="Number of training steps per epoch. Default: 2000"
    )
    parser.add_argument(
        "--eps_const", 
        type=float, 
        default=0.05, 
        help="Proportion of time to keep epsilon constant initially. Default: 0.05"
    )
    parser.add_argument(
        "--eps_max", 
        type=float, 
        default=0.95, 
        help="Determines the final frame (as a percentage of total frames) to degrade epsilon. Default: 0.95"
    )
    parser.add_argument(
        "--epsilon_start", 
        type=float, 
        default=1.0, 
        help="Epsilon start value. Default: 1.0"
    )
    parser.add_argument(
        "--epsilon_final", 
        type=float, 
        default=0.01, 
        help="Epsilon final value. Default: 0.01"
    )
    parser.add_argument(
        "--replay_buffer_size", 
        type=int, 
        default=200000, 
        help="Size of the replay buffer. Default: 500000"
    )
    parser.add_argument(
        "--initialise_buffer_percent", 
        type=int, 
        default=50, 
        help="Percentage of buffer to populate with random actions before training"
    )
    parser.add_argument(
        "--initialise_buffer", 
        type=str, 
        default="", 
        help="If init initialises buffer and saves only, if load it loads the buffer from file, \
            otherwise initialises buffer and continues"
    )
    parser.add_argument(
        "--buffer_skip", 
        type=int, 
        default=2, 
        help="If initialise_buffer is init, only records every n experiences in the buffer. Default: 2"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size for network training. Default: 32"
    )
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=0.99, 
        help="Discount factor to apply to previous rewards. Default: 0.99"
    )
    parser.add_argument(
        "--model_save_interval", 
        type=int, 
        default=10000, 
        help="Save the model after every X epochs. Default: 10"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="./runs", 
        help="Directory to write tensorboard logs to. Default: ./runs"
    )
    parser.add_argument(
        "--troubleshoot", 
        type=bool, 
        default=False, 
        help="for troubleshooting the code. Trains with /n window visible, 2 epochs, \
        100 steps per epoch, sync networks after 10 steps, replay buffer size of 100, batch size of 3, log to ./troubleshooting_runs"
    )

    args = parser.parse_args()

    if args.troubleshoot:
        args.epochs = 2
        args.steps_per_epoch = 1000
        args.sync_networks_after = 10
        args.replay_buffer_size = 100
        args.log_dir = './troubleshooting_runs'
        args.batch_size = 3

    device = torch.device("cuda" if args.gpu else "cpu")

    game = ExtendedGame(args.game_config_file_path)
    game.set_window_visible(args.window_visible)
    game.init() 
    writer = SummaryWriter(logdir=args.log_dir)
        
    total_training_steps = args.epochs * args.steps_per_epoch
    epsilon_decay_first_frame = total_training_steps * args.eps_const
    epsilon_decay_last_frame = total_training_steps * args.eps_max
    epsilon_amount_to_decay = args.epsilon_start - args.epsilon_final
    epsilon_decay_over_frames = epsilon_decay_last_frame - epsilon_decay_first_frame
    epsilon_decay_per_frame = epsilon_amount_to_decay / epsilon_decay_over_frames

    experience_buffer = ExperienceBuffer(
        args.replay_buffer_size, args.gamma
    )
    initialisation_steps = int(
        args.replay_buffer_size * args.initialise_buffer_percent / 100
    )

    # setup the training and target networks
    net_shape = (args.num_frames_stacked, args.width, args.height)
    train_net = ConvNet(net_shape, game.n_actions).to(device)
    target_net = ConvNet(net_shape, game.n_actions).to(device)

    agent = Agent(
        game, 
        train_net, 
        experience_buffer, 
        args.width, 
        args.height, 
        args.num_frames_stacked, 
        args.repeat_for_frames
    )

    epsilon = args.epsilon_start
    optimizer = optim.Adam(train_net.parameters(), lr=args.learning_rate)

    last_100_rewards = collections.deque(maxlen=100)
    
    # to track the mean across epochs
    total_mean = 0
    total_count = 0
    
    dt = datetime.datetime.now()
    date_time_string = dt.strftime("%Y%m%d-%H-%M-%S")
    model_path = 'models/' + date_time_string
    os.makedirs(model_path, mode=0o755, exist_ok=True)
    steps = 0
    last_reward_step = 0
    ts = time.time()
    
    if args.initialise_buffer == "init":
        steps = args.replay_buffer_size * args.buffer_skip
        print(
            "Initialising and saving buffer with %d transitions" 
            % args.replay_buffer_size
        )
        for step in trange(steps, leave=False):
            if (step % args.buffer_skip == 0):
                agent.play_step(True, device, write_to_buffer=True)
            else:
                agent.play_step(True, device, write_to_buffer=False)

        with open("experience_buffer.pickle", "wb") as f:
            pickle.dump(experience_buffer, f)
        sys.exit()

    elif args.initialise_buffer == "load":
        print("loading buffer ...")
        with open("experience_buffer.pickle", "rb") as f:
            experience_buffer = pickle.load(f)
            print("buffer size: {}".format(len(experience_buffer)))
            
    else:
        print("Initialising buffer with %d transitions" % initialisation_steps)
        for step in trange(initialisation_steps, leave=False):
            agent.play_step(True, device)
    
    for epoch in range(args.epochs):
        rewards = []

        print("\n-------- Epoch %d\n--------" % (epoch + 1))
        for step in trange(args.steps_per_epoch, leave=False):
            # epsilon is constant for some steps and then decays
            steps += 1
            if steps >= epsilon_decay_first_frame:
                epsilon = max( args.epsilon_final, (epsilon - epsilon_decay_per_frame) )
            else:
                epsilon = args.epsilon_start
            
            if np.random.random() < epsilon:
                done_reward = agent.play_step(True, device)
            else:
                done_reward = agent.play_step(False, device)
            
            if len(experience_buffer) >= args.replay_buffer_size:

                # periodically sync the training and target network weights
                if steps % args.sync_networks_after == 0:
                    target_net.load_state_dict(train_net.state_dict())

                # update weights
                optimizer.zero_grad()
                batch = experience_buffer.sample(args.batch_size)
                loss_t = calc_loss(batch, train_net, target_net, args.gamma, device=device)
                loss_t.backward()
                optimizer.step()
                writer.add_scalar("loss", loss_t, steps)
            
            if done_reward:
                rewards.append(done_reward)
                last_100_rewards.append(done_reward)

                # generate and write some stats for tensorboard
                speed = (steps - last_reward_step) / (time.time() - ts)
                last_reward_step = steps
                ts = time.time()
                writer.add_scalar("epsilon", epsilon, steps)
                writer.add_scalar("speed", speed, steps)
                writer.add_scalar("reward", done_reward, steps)
                writer.add_scalar("mean_100_reward", np.mean(last_100_rewards), steps)
            
        e_mean = np.mean(rewards)
        e_std = np.std(rewards)
        e_min = np.min(rewards)
        e_max = np.max(rewards)
        total_mean += e_mean
        total_count += 1
        overall_mean = np.mean(total_mean/total_count)
        print("\nOverall mean reward %d epoch mean %d, epoch STD %d, epoch min %d, epoch max %d" % (overall_mean, e_mean, e_std, e_min, e_max) )
        
        if (epoch % args.model_save_interval) == 0:
            model_file = model_path + '/' + str(epoch) + '.pth'
            torch.save(train_net.state_dict(), model_file)
            print('\nSaved model to ' + model_file)
    
    writer.close()