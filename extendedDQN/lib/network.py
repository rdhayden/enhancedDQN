import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import math

# Adapted from "Deep Reinforcement Learning Hands-On", Maxim Lapan, Packt Publishing, June 2018, ISBN:978-1-78883-424-7
class NoisyFactorizedLinear(nn.Linear):
    """
    Noisy network layer with factorized gaussian noise
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        return functional.linear(input, self.weight + self.sigma_weight * noise_v, bias)

class Network(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Network, self).__init__()

        # convolution layers generate features from video frames
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        # get input shape for the fully connected layer once at initialisation
        conv_out_size = self._get_conv_out(input_shape)

        # instantiate noisy layers, first 2 for the state_value and the last 2 for the advantage
        self.noisy_layers = [
            NoisyFactorizedLinear(conv_out_size, 512),
            NoisyFactorizedLinear(512, 1),
            NoisyFactorizedLinear(conv_out_size, 512),
            NoisyFactorizedLinear(512, n_actions),
        ]

        # fully connected layers for the state value
        self.fc_state_value = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

        # fully connected layers for the advantage
        self.fc_advantage = nn.Sequential(
            self.noisy_layers[2],
            nn.ReLU(),
            self.noisy_layers[3]
        )

    def _get_conv_out(self, shape):
        """Utility to get the output size of the convolution layers"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """process a batch forward through the network"""
        conv_out = self.conv(x).view(x.size()[0], -1)
        state_value = self.fc_state_value(conv_out)
        advantage = self.fc_advantage(conv_out)
        return state_value + advantage - advantage.mean()

    # to get signal to noise ratio for logging
    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]
