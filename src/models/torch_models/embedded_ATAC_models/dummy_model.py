import os
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import torch_model_construction_utils as tmu

num_factors = 19  
batch_size = 128
embedded_seq_len = 84300
embedding_dim_len = 300

default_hyperparams={'first_filters': 30,
                     'weight_lambda': 5e-3,
                     'bias_lambda': 5e-3,
                     'sparse_lambda': 10e-3}

train_loss = nn.BCEWithLogitsLoss(size_average=False)
valid_loss = nn.BCEWithLogitsLoss(size_average=False)
data_cast = lambda x: torch.autograd.Variable(x).float()
label_cast = lambda y: torch.autograd.Variable(y).float()

class BindSpaceNet(nn.Module):

    def __init__(self, input_size=(1, 281, 300), num_factors=24, hyperparams_dict=default_hyperparams):

        super(BindSpaceNet, self).__init__()
        self.relu = nn.SELU()
        num_filters = hyperparams_dict['first_filters']
        
        # shared 'palm' parameters
        self.conv1 = nn.Conv2d(1, num_filters, (1,300))
        self.pool1 = nn.MaxPool2d(kernel_size=(3,1)) 

        # fixture parameters to calculate flattened represenation 
        self.conv2 = nn.Conv2d(num_filters,10,(30,1))
        self.pool2 = nn.MaxPool2d((4,1))
        conv_size = self._get_conv_output(input_size)  

        # factor specific 'finger' parameters
        self.fingers = nn.ModuleList([nn.Sequential(nn.Conv2d(num_filters,10,(30,1)),
                                            self.relu,
                                            nn.MaxPool2d((4,1)),
                                            nn.Dropout(p=0.25)) for f in range(num_factors)])
        self.tips = nn.ModuleList([nn.Linear(conv_size, 1) for f in range(num_factors)])
        

    def forward(self, input):

        # shared forward computation 
        x = self.pool1(self.relu(self.conv1(input)))
        
        # factor-specific forward computations
        fingers = [finger(x) for finger in self.fingers]
        
        # flatten the output of each factor specific computations        
        flattened_fingers = [f.view(f.size(0), -1) for f in fingers]
        
        # calculate the output of each factor specific layer
        factor_outputs = [tip(f) for tip, f in zip(self.tips, flattened_fingers)]

        return sum(factor_outputs)

    # helper function to calculate number of units to expect for 
    # FC layers
    def _get_conv_output(self, shape):
        bs = 1
        fixture_input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(fixture_input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x_c1 = self.conv1(x)
        x_p1 = self.pool1(x_c1)
        x_c2 = self.conv2(x_p1)
        x_p2 = self.pool2(x_c2)
        return x_p2

net = BindSpaceNet(num_factors=num_factors, hyperparams_dict=default_hyperparams)
# Initialize model params
net.apply(tmu.init_weights)

# Collect weight, bias parameters for regularization
weights, biases, sparse_weights = tmu.get_model_param_lists(net)
