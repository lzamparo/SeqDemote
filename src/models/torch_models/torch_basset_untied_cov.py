import os
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class Conv2dUntiedBias(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_len, stride=1, padding=0, dilation=1, groups=1):
        super(Conv2dUntiedBias, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        
        height = 1
        width = self.calc_output_width(input_len, kernel_size)
        self.bias = nn.Parameter(torch.Tensor(out_channels, height, width))
        self.reset_parameters()

    def calc_output_width(self, input_length, kernel_size, stride=1):
        return (input_length - kernel_size[-1] + stride) // stride

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = F.conv2d(input, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)
        # add untied bias
        output += self.bias.unsqueeze(0).repeat(input.size(0), 1, 1, 1)
        return output
        

from load_pytorch import DNase_Train_Dataset, DNase_Valid_Dataset

data_path = os.path.expanduser("~/projects/SeqDemote/data/DNase/encode_roadmap.h5")
save_dir = "basset_repro_pytorch"

batch_size = 128
momentum = 0.98
clipping = 7
cuda = True

learning_rate_schedule = {
0: 0.003,
10: 0.002,
20: 0.0001}

validate_every = 1
save_every = 1

train_loss = nn.BCELoss()
valid_loss = nn.BCELoss()
train_dataset = DNase_Train_Dataset(data_path, transform=None)
valid_dataset = DNase_Valid_Dataset(data_path, transform=None)
data_cast = lambda x: torch.autograd.Variable(x).float()
label_cast = lambda y: torch.autograd.Variable(y).float()


class BassetRepro(nn.Module):
    
    def __init__(self, input_size=(4,1,600)):
        
        super(BassetRepro, self).__init__()
        self.relu = nn.ReLU()
        
        self.conv1 = Conv2dUntiedBias(4, 300, (1,19), input_size[-1])
        self.bn1 = nn.BatchNorm2d(300)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,3), stride=(1,3))
        
        output_width = self.conv1.calc_output_width(input_size[-1],kernel_size=(1,19))
        input_width = output_width // 3
        
        self.conv2 = Conv2dUntiedBias(300, 200, (1,11), input_width)
        self.bn2 = nn.BatchNorm2d(200)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))
        
        output_width = self.conv2.calc_output_width(input_width,kernel_size=(1,11))
        input_width = output_width // 4
        
        self.conv3 = Conv2dUntiedBias(200, 200, (1,7), input_width)
        self.bn3 = nn.BatchNorm2d(200)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))
        
        output_width = self.conv3.calc_output_width(input_width,kernel_size=(1,7))
        input_width = output_width // 4
        
        conv_size = self._get_conv_output(input_size)
        
        self.fc1 = nn.Linear(conv_size, 1000)
        self.bn4 = nn.BatchNorm1d(1000)
        self.drop1 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(1000,1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.drop2 = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(1000,164)
        self.out_layer = nn.Sigmoid()
        
        
    def forward(self, input):
        
        x = self.pool1(self.relu(self.bn1(self.conv1(input))))
        
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        # flatten layer
        x = x.view(x.size(0), -1)
        
        x = self.drop1(self.relu(self.bn4(self.fc1(x))))
        
        x = self.drop2(self.relu(self.bn5(self.fc2(x))))
        
        x = self.fc3(x)
        
        return self.out_layer(x)
        
    
    
    # helper function to calculate number of units to expect for 
    # FC layers
    def _get_conv_output(self, shape):
        bs = 1
        fixture_input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat, _ = self._forward_features(fixture_input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x_c1 = self.conv1(x)
        y = F.relu(F.max_pool2d(x_c1, (1,3), stride=(1,3)))
        y_c2 = self.conv2(y)
        z = F.relu(F.max_pool2d(y_c2, (1,4), stride=(1,4)))
        z_c3 = self.conv3(z)
        final = F.relu(F.max_pool2d(z_c3, (1,4), stride=(1,4)))
        convout_sizes = [c.size()[2:] for c in [x_c1, y_c2, z_c3]]
        return final, convout_sizes
    
net = BassetRepro()


def init_weights(m, gain=nn.init.calculate_gain('relu')):
    ''' Recursively initalizes the weights of a network. '''
    
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    
    if isinstance(m, nn.Conv2d):
        torch.nn.init.orthogonal_(m.weight, gain)
        m.bias.data.fill_(0.1)

net.apply(init_weights)

weights, biases = [], []
for name, p in net.named_parameters():
    if 'bias' in name:
        biases += [p]
        
    else:
        weights += [p]
  
  
# Initialize the params        
optimizer = torch.optim.RMSprop
optimizer_param_dicts = [
    {'params': weights, 'weight_decay': 5e-3},
        {'params': biases, 'weight_decay': 5e-3}
]
optimizer_kwargs = {'lr': learning_rate_schedule[0], 'momentum': momentum}
