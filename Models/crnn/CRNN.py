"""
MRI Convolutional Recurrent Neural Network.

Author: David Wilson. 2019.
Email: david.wilson7@outlook.com

References:
"Convolutional Recurrent Neural Networks for Dynamic MR Image Reconstruction"
Chen Qin, Jo Schlemper, Jose Caballero, Anthony Price, Joseph V. Hajnal, Daniel Rueckert
"""
# Import packages needed
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io

# Import packages for PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

# For using cuda GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def data_consistency(k, k_0, mask, tau=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    # This is the regualrisation parameter
    # lambda and tau are same thing
    lambda = tau
    
    if lambda:  
        output = (1 - mask) * k + mask * (k + lambda * k_0) / (1 + lambda)
    else:  
        output = (1 - mask) * k + mask * k_0
    return output

class Data_Consistency(nn.Module):
    """
    Data Consistency layer
    See the reference for the details of this operation.
    The basic idea is that if a point is smapled, we take the linear combination
    between the CNN prediction and the original measurements, weighted by the levle of noise.
    If the point is not sampled then we use the output of the network.
    """

    def __init__(self, tau=None, norm='ortho'):
        super(Data_Consistency, self).__init__()
        self.tau = tau
        self.normalized = norm == 'ortho' # needed for FFT

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k_0, mask):
        """
        x: input in the image domain, shape: (N, 2, x, y, T) 
                             {batch_size, channel_size, width, height, time-steps}
        k_0: initially sampled k-space data
        mask: the mask we use which tells which points were sampled which were not
        """
        #2D data
        if x.dim() == 4:
            x    = x.permute(0, 2, 3, 1)
            k_0   = k_0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
        #3D data    
        elif x.dim() == 5: 
            x    = x.permute(0, 4, 2, 3, 1)
            k_0   = k_0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)

        # See the paper for more information
        # F^T * diagoanl matrix * F * x
        k = torch.fft(x, 2, normalized=self.normalized)
        output = data_consistency(k, k_0, mask, self.tau)
        output = torch.ifft(output, 2, normalized=self.normalized)

        if x.dim() == 4:
            output = output.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            output = output.permute(0, 4, 2, 3, 1)

        return output

class CRNN(nn.Module):
    """
    Convolutional recurrent units evolving over iterations only
    Parameters
    -----------------
    input = input of the cell, 4D tensor, shape: (N_batch, channel, width, height)
    hiddenIteration = hidden states in the iteration dimension, 4D tensor, shape: (N_batch, hidden_channel, width, height)
    hiddenTempral = hidden states in the tmeporal dimension, 4D tensor, shape: (N_batch, hidden_channel, width, height)
    -----------------
    output ---> hidden = our hidden representation, 4D tensor, shape: (N_batch, hidden_channel, width, height)
    """

    def __init__(self, input_channel, hidden_channel, kernel_size):
        super(CRNN, self).__init__()
        self.kernel_size = kernel_size
        self.iteration_to_hidden = nn.Conv2d(input_channel, hidden_channel, kernel_size, padding=1)
        self.hidden_to_hidden = nn.Conv2d(hidden_channel, hidden_channel, kernel_size, padding=1)
        # add iteration hidden connection
        self.iterationHidden_to_iterationHidden = nn.Conv2d(hidden_channel, hidden_channel, kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(hidden_channel)

    def forward(self, input, hiddenIteration, hiddenTemporal):
        """
        This performs the following operation, giving out H_l^i which is the
        hidden representation at layer l and at iteration i.
        ---------------
        H_l^i = ReLU(W_l * H_{l-1}^i + W_i * H_l^{i-1} + B_l)
        where ReLU is the non-linearity
        W_l and W_i are the filters of input_to_hidden convolutions and hidden_to_hidden convolutions respectively
        B_l is the bias term
        * is Convolution operation
        ---------------
        Also, output is batch-normalised at the end
        """
        input_to_hidden = self.iteration_to_hidden(input)
        hidden_to_hidden = self.hidden_to_hidden(hiddenTemporal)
        iterationHidden = self.iterationHidden_to_iterationHidden(hiddenIteration)

        hidden = self.relu(input_to_hidden + hidden_to_hidden + iterationHidden)
        #print(hidden.shape)
        hidden = self.batchnorm(hidden)

        return hidden

class BCRNN(nn.Module):
    """
    Bidirectional convolutional recurrent units evolving over time and iterations
    ---------------------
    input = input data, 5D tensor, shape: (time_steps-T, N_batch, channel, width, height)
    inputIteration = hidden states form the previous iteration, 5D tensor, shape: (time_steps-T, N_batch, channel, width, height)
    mode = If in test mode to remove the grad. False (training) or True (testing)
    ---------------------
    output---> output,
    """
    def __init__(self, input_channel, hidden_channel, kernel_size):
        super(BCRNN, self).__init__()
        self.hidden_channel = hidden_channel
        self.kernel_size = kernel_size
        self.input_channel = input_channel
        self.CRNN_model = CRNN(self.input_channel, self.hidden_channel, self.kernel_size)

    def forward(self, input, input_iteration, mode=False):
        """
        This performs the following operation, giving out H_l_t^i which is the
        hidden representation at layer l, time-step t and at iteration i.
        ---------------
        forward_H_l_t^i = ReLU(W_l * H_{l-1}_t^i + W_t * forward_H_l_{t-1}^{i}+ W_i * H_l_t^{i-1} + forward_B_l)
        backward_H_l_t^i = ReLU(W_l * H_{l-1}_t^i + W_t * backward_H_l_{t-1}^{i}+ W_i * H_l_t^{i-1} + backward_B_l)
        H_l_t^i = forward_H_l_t^i + backward_H_l_t^i
        where ReLU is the non-linearity
        B_l is the bias term
        * is Convolution operation
        W_l and W_i are the filters of input_to_hidden convolutions and hidden_to_hidden convolutions respectively
        and W_t represents the filters of recurrent convolutions evolving over time.
        forward and backward is the direction of the hidden representation.
        ---------------
        """

        T, N_batch, channels, X, Y = input.shape # T, N_batch, channels, x, y
        hidden_size = [N_batch, self.hidden_channel, X, Y]
        
        if mode: # If testing
            with torch.no_grad():
                initial_hidden = Variable(torch.zeros(hidden_size)).cuda()
        else: # If training
            initial_hidden = Variable(torch.zeros(hidden_size)).cuda()

        forward = [] # Forward representation
        backward = [] # Forward representation
        
        # forward pass in time
        hidden = initial_hidden
        for i in range(T):
            hidden = self.CRNN_model(input[i], input_iteration[i], hidden)
            forward.append(hidden)

        forward = torch.cat(forward)

        # backward pass in time
        hidden = initial_hidden
        for i in range(T):
            hidden = self.CRNN_model(input[T - i - 1], input_iteration[T - i -1], hidden)

            backward.append(hidden)
        backward = torch.cat(backward[::-1])
        
        # The output representation which is the sum of 
        # forward and the backward pass in time
        output = forward + backward

        if N_batch == 1:
            output = output.view(T, 1, self.hidden_channel, X, Y)

        return output


class CRNN_MRI(nn.Module):
    """
    Model for Dynamic MRI Reconstruction using Convolutional Neural Networks
    Parameters. See the paper for more information
    -----------------------
         N_channels: number of channels
         N_filters: number of filters
         kernel_size: kernel size
         N_iterations: number of iterations
         N_units: number of CRNN/BCRNN/CNN layers in each iteration
    """
    def __init__(self, N_channels=2, N_filters=64, kernel_size=3, N_iterations=10, N_units=5):

        super(CRNN_MRI, self).__init__()
        self.N_iterations = N_iterations
        self.N_units = N_units
        self.N_filters = N_filters
        self.kernel_size = kernel_size

        self.bcrnn = BCRNN(N_channels, N_filters, kernel_size)
        self.conv1_x = nn.Conv2d(N_filters, N_filters, kernel_size, padding = 1) 
        self.conv1_h = nn.Conv2d(N_filters, N_filters, kernel_size, padding = 1)

        self.conv2_x = nn.Conv2d(N_filters, N_filters, kernel_size, padding = 1)
        self.conv2_h = nn.Conv2d(N_filters, N_filters, kernel_size, padding = 1)

        self.conv3_x = nn.Conv2d(N_filters, N_filters, kernel_size, padding = 1)
        self.conv3_h = nn.Conv2d(N_filters, N_filters, kernel_size, padding = 1)

        self.conv4_x = nn.Conv2d(N_filters, N_channels, kernel_size, padding = 1)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(N_filters)
        self.batchnorm3 = nn.BatchNorm3d(N_channels)

        DC = []
        for i in range(N_iterations):
            DC.append(Data_Consistency(norm='ortho', tau=0.2))
        self.DC = DC

    def forward(self, input, k, m, mode=False):
        """
        input, y, m: input image, k-spce data, mask, shape: (N_batch, 2, X, Y, T)
        note that the input and x are equivalent, so conv_x is the convolution from input
        mode - True: the model is in test mode, False: train mode
        """
        network = {} # The network is a dictionary
        n_batch, N_channels, width, height, n_seq = input.size()
        size_h = [n_seq * n_batch, self.N_filters, width, height]
        if mode:
            with torch.no_grad():
                initial_hidden = Variable(torch.zeros(size_h)).cuda()
        else:
            initial_hidden = Variable(torch.zeros(size_h)).cuda()

        for j in range(self.N_units-1):
            network['T0_X%d'%j] = initial_hidden

        for i in range(1, self.N_iterations+1):

            input = input.permute(4,0,1,2,3)
            input = input.contiguous()
            
            network['T%d_X0' %(i-1)] = network['T%d_X0' % (i - 1)].view(n_seq, n_batch,self.N_filters,width, height)
            # Applying the BCRNN
            network['T%d_X0' %i] = self.bcrnn(input, network['T%d_X0'%(i-1)], mode)
            network['T%d_X0' %i] = network['T%d_X0'%i].view(-1,self.N_filters,width, height)
            # First CRNN
            network['T%d_X1' %i] = self.conv1_x(network['T%d_X0'%i])
            network['T%d_H1' %i] = self.conv1_h(network['T%d_X1'%(i-1)])
            network['T%d_X1' %i] = self.relu(network['T%d_H1'%i] + network['T%d_X1'%i])
            network['T%d_X1' %i] = self.batchnorm(network['t%d_x1'%i])
            # Second CRNN
            network['T%d_X2' %i] = self.conv2_x(network['T%d_X1'%i])
            network['T%d_H2' %i] = self.conv2_h(network['T%d_X2'%(i-1)])
            network['T%d_X2' %i] = self.relu(network['T%d_H2'%i] + network['T%d_X2'%i])
            network['T%d_X2' %i] = self.batchnorm(network['T%d_X2'%i])
            # Third CRNN
            network['T%d_X3'%i] = self.conv3_x(network['T%d_X2'%i])
            network['T%d_X3'%i] = self.conv3_h(network['T%d_X3'%(i-1)])
            network['T%d_X3'%i] = self.relu(network['T%d_H3'%i] + network['T%d_X3'%i])
            network['T%d_X3'%i] = self.batchnorm(network['T%d_X3'%i])
            # Last CNN
            network['T%d_X4'%i] = self.conv4_x(network['T%d_X3'%i])

            input = input.view(-1,N_channels,width, height)
            # Skip connection
            network['T%d_out'%i] = input + network['T%d_X4'%i]

            network['T%d_OUTPUT'%i] = network['T%d_OUTPUT'%i].view(-1,n_batch, N_channels, width, height)
            network['T%d_OUTPUT'%i] = network['T%d_OUTPUT'%i].permute(1,2,3,4,0)
            network['T%d_OUTPUT'%i].contiguous()
            # Data Consistency
            network['T%d_OUTPUT'%i] = self.DC[i-1].perform(network['T%d_OUTPUT'%i], k, m)
            input = network['T%d_OUTPUT'%i]
            #input = self.batchnorm3(input)

            # In test mode, cleans up all previous recurrent iterations
            if mode:
                to_delete = [key for key in network if ('T%d'%(i-1)) in key]

                for deletion in to_delete:
                    del network[deletion]

                torch.cuda.empty_cache()

        return network['T%d_OUTPUT'%i]
