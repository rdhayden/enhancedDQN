import torch
import torch.nn as nn
import numpy as np

# Taken from "Deep Reinforcement Learning Hands-On", Maxim Lapan, Packt Publishing, June 2018, ISBN:978-1-78883-424-7
class ConvNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ConvNet, self).__init__()

        # convolution layers generate features from video frames
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        # get input shape for the fully connected layer once at initialisation
        conv_out_size = self._get_conv_out(input_shape)

        # fully connected layers generates best action
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            #nn.ReLU(),
            #nn.Linear(1024, 512),
            #nn.ReLU(),
            #nn.Linear(512, n_actions)
            nn.Linear(1024, n_actions)
        )

    def _get_conv_out(self, shape):
        """Utility to get the output size of the convolution layers"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """process a batch forward through the network"""
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
