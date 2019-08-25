"""
Residual U-net for De-aliasing MR images.
Author: David Wilson. 2019
Email: david.wilson7@outlook.com
References:
"""
# Importing main packages
import numpy as np
import h5py
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# For using cuda GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class unet(nn.Module):
    """
    Implementation of the U-net as in the Hauptmann et al.
    --------------
    inputs = Aliased data, 4D tensor,
            shape: (N_batch, N_channels, depth, height, width)
    The order of the depth, height and width doesn't really matter.
    --------------
    outputs---> De-aliased data, 4D tensor,
            shape: (N_batch, N_channels, depth, height, width)
    """
    def __init__(self, input_channel=1, first_output_channel=32, kernel_size=3):
        super(unet, self).__init__()
        self.kernel_size = kernel_size
        # All the convolutions used.
        # The subscript is the level going down until to 3
        # and then goes upwards
        # convi_i is the same convolution as i repeated but differnet
        # input channels

        self.conv1 = nn.Conv3d(input_channel, first_output_channel,
                               kernel_size, padding=1)
        # Default: self.conv1 = nn.Conv3d(1, 32, kernel_size, padding=1)
        self.conv1_1 = nn.Conv3d(first_output_channel, first_output_channel,
                                 kernel_size, padding=1)
        # Notice the differnece in the input channel compared to previous one

        self.conv2 = nn.Conv3d(first_output_channel, first_output_channel*2,
                               kernel_size, padding=1)
        # Default: self.conv2 = nn.Conv3d(32, 64, kernel_size, padding=1)
        self.conv2_2 = nn.Conv3d(first_output_channel*2, first_output_channel*2,
                                 kernel_size, padding=1)

        self.conv3 = nn.Conv3d(first_output_channel*2, first_output_channel*2*2,
                               kernel_size, padding=1)
        # Default: self.conv3 = nn.Conv3d(64, 128, kernel_size, padding=1)
        self.conv3_3 = nn.Conv3d(128, 128, kernel_size, padding=1)

        self.conv4 = nn.Conv3d(first_output_channel*2*2, first_output_channel*2,
                               kernel_size, padding=1)
        # Default: self.conv4 = nn.Conv3d(128, 64, kernel_size, padding=1)
        self.conv4_4 = nn.Conv3d(first_output_channel*2, first_output_channel*2,
                                 kernel_size, padding=1)

        self.conv5 = nn.Conv3d(first_output_channel*2, first_output_channel,
                               kernel_size, padding=1)
        # Default: self.conv5 = nn.Conv3d(64, 32, kernel_size, padding=1)
        self.conv5_5 = nn.Conv3d(first_output_channel, first_output_channel,
                                 kernel_size, padding=1)

        self.conv6 = nn.Conv3d(first_output_channel, input_channel,
                               kernel_size, padding=1)
        # Default: self.conv6 = nn.Conv3d(32, 1 , kernel_size, padding=1)

        # Transposed convolutions for upsampling
        self.convT1 = nn.ConvTranspose3d(first_output_channel*2*2,
                                         first_output_channel*2, kernel_size,
                                         stride=(2,2,2), padding=1,
                                         output_padding=1)
        # Default: self.convT1 = nn.ConvTranspose3d(128, 64, kernel_size,
        #                        stride=(2,2,2), padding=1, output_padding=1)
        self.convT2 = nn.ConvTranspose3d(first_output_channel*2,
                                         first_output_channel,
                                         kernel_size, stride=(2,2,2),
                                         padding=1, output_padding=1)
        # Default: self.convT2 = nn.ConvTranspose3d(64, 32, kernel_size,
        #                          stride=(2,2,2), padding=1, output_padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        """
        1- Convolution, Convolution, pooling
        2- Convolution, Convolution, pooling
        3- Convolution, Convolution, upsampling
        4- Convolution, Convolution, upsampling
        5- Convolution, Convolution, Convolution
        6- Skip connection, Relu
        """
        conv1 = self.relu(self.conv1(inputs))
        conv1 = self.relu(self.conv1_1(conv1))
        pool1 = F.max_pool3d(conv1, 2)

        conv2 = self.relu(self.conv2(pool1))
        conv2 = self.relu(self.conv2_2(conv2))
        pool2 = F.max_pool3d(conv2, 2)

        conv3 = self.relu(self.conv3(pool2))
        conv3 = self.relu(self.conv3_3(conv3))
        conv3 = self.convT1(conv3)

        up1 = torch.cat((conv3, conv2), dim=1)

        conv4 = self.relu(self.conv4(up1))
        conv4 = self.relu(self.conv4_4(conv4))

        conv4 = self.convT2(conv4)

        up2 = torch.cat((conv4, conv1), dim=1)

        conv5 = self.relu(self.conv5(up2))
        conv5 = self.relu(self.conv5_5(conv5))

        conv6 = self.relu(self.conv6(conv5) + inputs)
        # conv6 is the output
        return conv6

