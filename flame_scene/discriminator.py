import os
import torch
import numpy as np
from torch import nn


class Discriminator_512(nn.Module):
    def __init__(self, input_feature,output_feature):
        super(Discriminator_512, self).__init__()
        self.main = nn.Sequential(
            
            nn.Conv2d(input_feature, output_feature, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(output_feature, output_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_feature * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(output_feature * 2, output_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_feature * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(output_feature * 4, output_feature * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_feature * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(output_feature * 8, output_feature * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_feature * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(4096*output_feature,512*output_feature),
            nn.ReLU(),
            nn.Linear(512*output_feature, 64*output_feature),
            nn.ReLU(),
            nn.Linear(64 * output_feature, 8 * output_feature),
            nn.ReLU(),
            nn.Linear(8 * output_feature, output_feature),
            nn.ReLU(),
            nn.Linear(output_feature, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator_new(nn.Module):
    def __init__(self, input_feature, output_feature):
        super(Discriminator_new, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(input_feature, output_feature, 4, 2, 0, bias=False),#400 274
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(output_feature, output_feature * 2, 4, 2, (1,0), bias=False),#200 136
            nn.BatchNorm2d(output_feature * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(output_feature * 2, output_feature * 4, 4, 2, 1, bias=False),#100 68
            nn.BatchNorm2d(output_feature * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(output_feature * 4, output_feature * 8, 4, 2, 1, bias=False),#50 34
            nn.BatchNorm2d(output_feature * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(output_feature * 8, output_feature * 16, 4, 2, (1,0), bias=False),#25 16
            nn.BatchNorm2d(output_feature * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(51200, 5120),
            nn.ReLU(),
            nn.Linear(5120, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
