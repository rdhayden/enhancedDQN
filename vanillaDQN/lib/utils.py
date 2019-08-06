import torch
import torch.nn as nn

# loss generator
def calc_loss(batch, train_net, target_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    
    states_v = torch.tensor(states, dtype=torch.float).to(device)
    next_states_v = torch.tensor(next_states, dtype=torch.float).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)

    # done's are Booleans, so the tensor can be used as a mask
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = train_net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = target_net(next_states_v).max(1)[0]

    next_state_values[done_mask] = 0.0

    # detach the next state values so that back propagation does not update the target network
    next_state_values = next_state_values.detach()

    expected_state_action_values = rewards_v + gamma * next_state_values
    return nn.MSELoss()(state_action_values, expected_state_action_values)