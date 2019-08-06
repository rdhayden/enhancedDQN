import pickle
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
from tensorboardX import SummaryWriter
import argparse
import datetime
import os, sys

if __name__ == "__main__":

    start_date_time = datetime.datetime.now()

    parser = argparse.ArgumentParser(description="Trains Enhanced ViZDoom agent")
    parser.add_argument(
        "--window_visible",
        type=bool,
        default=False,
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
        default="scenarios/basic.cfg",
        help="Path to game config file. Default: scenarios/basic.cfg",
    )
    parser.add_argument(
        "--repeat_for_frames",
        type=int,
        default=4,
        help="Number of frames to repeat each action. Default: 4",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00001,
        help="Learning rate for the optimizer. Default: ",
    )
    parser.add_argument(
        "--sync_networks_after",
        type=int,
        default=500,
        help="Steps between synchronization of target and training networks. Default: 500",
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
        "--replay_buffer_size",
        type=int,
        default=500000,
        help="Size of the replay buffer. Default: 500000",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2,
        help="Number of steps for n_step look ahead. Default: 2",
    )
    parser.add_argument(
        "--initialise_buffer_percent",
        type=int,
        default=50,
        help="Percentage of buffer to populate with random actions before training",
    )
    parser.add_argument(
        "--initialise_buffer",
        type=str,
        default="",
        help="If init, initialises buffer and saves it only, if load it loads the buffer from file, otherwise initialises buffer and continues",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for network training. Default: 32",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor to apply to previous rewards. Default: 0.99",
    )
    parser.add_argument(
        "--model_save_interval",
        type=int,
        default=10000,
        help="Save the model after every X epochs. Default: 10",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./runs",
        help="Directory to write tensorboard logs to. Default: ./runs",
    )
    parser.add_argument(
        "--troubleshoot",
        type=bool,
        default=False,
        help="for troubleshooting the code. Trains for 2 epochs, \
        100 steps per epoch, sync networks after 10 steps, replay buffer size of 100, batch size of 3, log to ./troubleshooting_runs",
    )
    parser.add_argument(
        "--health_reward",
        type=int,
        default=0,
        help="sets the reward for negative change in health. Default: 0",
    )

    args = parser.parse_args()

    if args.troubleshoot:
        args.epochs = 2
        args.steps_per_epoch = 100
        args.sync_networks_after = 10
        args.replay_buffer_size = 100
        args.log_dir = "./troubleshooting_runs"
        args.batch_size = 3

    device = torch.device("cuda" if args.gpu else "cpu")

    # Initialise the game instance
    game = ExtendedGame(args.game_config_file_path)
    game.set_window_visible(args.window_visible)
    game.init()
    writer = SummaryWriter(logdir=args.log_dir)

    # Setup the experience buffer
    experience_buffer = ExperienceBuffer(
        args.replay_buffer_size, args.n_steps, args.gamma
    )

    # The number of steps to initialise the buffer before starting training
    initialisation_steps = int(
        args.replay_buffer_size * args.initialise_buffer_percent / 100
    )

    # Setup the training and target networks
    net_shape = (args.num_frames_stacked, args.width, args.height)
    train_net = Network(net_shape, game.n_actions).to(device)
    target_net = Network(net_shape, game.n_actions).to(device)

    # Initialise the agent
    agent = Agent(
        game,
        train_net,
        experience_buffer,
        args.width,
        args.height,
        args.num_frames_stacked,
        args.repeat_for_frames,
        args.health_reward,
    )

    # Use Adam as the optimizer because its fast and it helps reduce the overall compute load
    optimizer = optim.Adam(train_net.parameters(), lr=args.learning_rate)

    last_100_rewards = collections.deque(maxlen=100)

    # to track the mean across epochs
    total_mean = 0
    total_count = 0

    dt = datetime.datetime.now()
    date_time_string = dt.strftime("%Y%m%d-%H-%M-%S")
    model_path = "models/" + date_time_string
    os.makedirs(model_path, mode=0o755, exist_ok=True)
    steps = 0
    last_reward_step = 0

    if args.initialise_buffer == "init":
        print(
            "Initialising and saving buffer with %d transitions"
            % args.replay_buffer_size
        )
        for step in trange(args.replay_buffer_size, leave=False):
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
            steps += 1
            done_reward = agent.play_step(False, device)

            if len(experience_buffer) >= args.replay_buffer_size:

                # periodically sync the training and target network weights
                if steps % args.sync_networks_after == 0:
                    target_net.load_state_dict(train_net.state_dict())

                # update weights
                optimizer.zero_grad()
                batch = experience_buffer.sample(args.batch_size)
                loss_t = calc_loss(
                    batch, train_net, target_net, args.gamma, device=device
                )
                loss_t.backward()
                optimizer.step()

            if done_reward:
                rewards.append(done_reward)
                last_100_rewards.append(done_reward)

                # generate and write some stats for tensorboard
                last_reward_step = steps
                writer.add_scalar("reward", done_reward, steps)
                writer.add_scalar("mean_100_reward", np.mean(last_100_rewards), steps)

            # write signal to noise stats for tensorboard
            if steps % 500 == 0:
                for layer_idx, weight_sigma_ratio in enumerate(train_net.noisy_layers_sigma_snr()):
                    writer.add_scalar("sigma_snr_layer_%d" % (layer_idx+1),
                                        weight_sigma_ratio, steps)

        # write some stats to stdout for each epoch
        e_mean = np.mean(rewards)
        e_std = np.std(rewards)
        e_min = np.min(rewards)
        e_max = np.max(rewards)
        total_mean += e_mean
        total_count += 1
        overall_mean = np.mean(total_mean / total_count)
        print(
            "\nOverall mean reward %d epoch mean %d, epoch STD %d, epoch min %d, epoch max %d"
            % (overall_mean, e_mean, e_std, e_min, e_max)
        )

        # periodically save the model
        if (epoch % args.model_save_interval) == 0:
            model_file = model_path + "/" + str(epoch) + ".pth"
            torch.save(train_net.state_dict(), model_file)
            print("\nSaved model to " + model_file)

    writer.close()

    finish_date_time = datetime.datetime.now()
    print("started: {}".format(start_date_time))
    print("finished: {}".format(finish_date_time))
