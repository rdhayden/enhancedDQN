# Deep Q learning for ViZDoom

## Prerequisites
vizdoom  <https://github.com/mwydmuch/ViZDoom>  
torch  
tqdm  
numpy  
opencV  
scikit-image  
absl-py  
tensorflow  
tensorboardx  
xxhash  
collections  
zlib  
may also need to install ffmpeg to play video on linux

## Arguments

The code supports many arguments to modify the way the agent is trained. To see the options and defaults:

python vd_learning.py -h

## Play pretrained model for the Enhanced DQN

*Visually plays 1 episode of model/deathAmmoPenalty.pth by default, use -h for more arguments*

python vd_play.py

## Training Examples for the Enhanced DQN

*To train the agent for 400000 steps (200 epochs and the default 2000 steps per epoch) with the defend the line scenario and a replay buffer of 200000, run the following. Also note that this configuration file provides a +1 reward for a kill and a -1 penalty for death*

python vd_learning.py --game_config_file_path "scenarios/defend_the_line_no_ammo_penalty.cfg" --epochs 200 --replay_buffer_size 200000 --log_dir runs/test200kill_1

*To add a penalty for loss of health of -1 whenever the agent is attacked to the above example, add the health_reward argument to the above as follows.*

python vd_learning.py --game_config_file_path "scenarios/defend_the_line_no_ammo_penalty.cfg" --epochs 200 --replay_buffer_size 200000 --health_reward -1 --log_dir runs/test200killhealth_1

*To train the agent for 400000 steps (200 epochs and the default 2000 steps per epoch) with the defend the line scenario and a replay buffer of 200000, run the following. Also note that this configuration file provides a +1 reward for a kill, -1 penalty for death and a -1 penalty for wasting ammo*

python vd_learning.py --game_config_file_path "scenarios/defend_the_line.cfg" --epochs 200 --replay_buffer_size 200000 --log_dir runs/test200killammo_1

*To add a health penalty of -1 whenever the agent is attacked to the previous example, so that it is penalised for death, health and wasting ammo, add the health_reward argument as follows.*

python vd_learning.py --game_config_file_path "scenarios/defend_the_line.cfg" --epochs 200 --replay_buffer_size 200000 --health_reward -1 --log_dir runs/test200all_1

## Training Examples for the Vanilla DQN

*To train the agent for 1000000 steps (500 epochs and the default 2000 steps per epoch) with the defend the line scenario and a replay buffer of 200000, run the following. Also note that this provides a +1 reward for a kill, and a -1 penalty for death and wasted ammo*

python vd_learning.py --game_config_file_path "scenarios/defend_the_line.cfg" --epochs 500 --replay_buffer_size 200000 --log_dir runs/test500all_1

## Code structure from the base directory:
.  
└-- extendedDQN                                    # contains all extended DQN code  
    |-- tests.py                                   # run this to perform tests  
    |-- vd_learning.py                             # run this to train the model  
    |-- vd_play.py                                 # play model -h for help  
    |-- README.md                                  # this readme  
    └-- lib  
        |-- __init.py__                            # does not include any initialisation code  
        |-- agent.py                               # agent takes actions, uses buffer etc.  
        |-- experience_buffer.py                   # the experience buffer  
        |-- extended_game.py                       # extended game to make actions with integers  
        |-- network.py                             # the deep neural networks  
        |-- utils.py                               # utility code to calculate the loss for back propagation  
    |-- models                                     # model files saved here  
    └-- scenarios  
        |-- basic.cfg                              # config for basic game scenario  
        |-- basic.wad                              # wad file (map etc) for basic game scenario  
        |-- defend_the_line_no_ammo_penalty.cfg    # config with +1 for kill and -1 for death  
        |-- defend_the_line_no_ammo_penalty.wad    # wad file with +1 for kill and -1 for death  
        |-- defend_the_line.cfg                    # config with +1 for kill, -1 for death & wasted ammo  
        |-- defend_the_line.wad                    # wad file with +1 for kill, -1 for death & wasted ammo  
└-- vanillaDQN                                     # contains all vanilla DQN code  
    |-- vd_learning.py                             # run this to train the model  
    |-- tests.py                                   # run this to perform tests  
    |-- README.md                                  # this readme  
    └-- lib  
        |-- __init.py__                            # does not include any initialisation code  
        |-- agent.py                               # agent takes actions, uses buffer etc.  
        |-- experience_buffer.py                   # the experience buffer  
        |-- extended_game.py                       # extended game to make actions with integers  
        |-- network.py                             # the deep neural networks  
        |-- utils.py                               # utility code to calculate the loss for back propagation  
    └-- models                                     # model files saved here  
    └-- scenarios  
        |-- basic.cfg                              # config for basic game scenario  
        |-- basic.wad                              # wad file (map etc) for basic game scenario  
        |-- defend_the_line_no_ammo_penalty.cfg    # config with +1 for kill and -1 for death  
        |-- defend_the_line_no_ammo_penalty.wad    # wad file with +1 for kill and -1 for death  
        |-- defend_the_line.cfg                    # config with +1 for kill, -1 for death & wasted ammo  
        └-- defend_the_line.wad                    # wad file with +1 for kill, -1 for death & wasted ammo  
