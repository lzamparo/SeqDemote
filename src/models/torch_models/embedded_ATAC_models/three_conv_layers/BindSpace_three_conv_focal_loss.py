import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from load_pytorch import Embedded_k562_ATAC_train_dataset, Embedded_k562_ATAC_validation_dataset
from load_pytorch import EmbeddingReshapeTransformer
from utils import torch_model_construction_utils as tmu

data_path = os.path.expanduser("~/projects/SeqDemote/data/ATAC/K562/K562_embed_TV_annotated_split.h5")
save_dir = "BindSpace_embedding_extension"

num_factors = 19  
batch_size = 128
momentum = None
embedded_seq_len = 84300
embedding_dim_len = 300
transformer = EmbeddingReshapeTransformer(embedding_dim_len, embedded_seq_len)
cuda = True

initial_lr = 0.005

model_hyperparams_dict={'gamma': {'type': 'float', 'min': 0.5, 'max': 3.0},
                        'alpha': {'type': 'float', 'min': 0.1, 'max': 0.9},
                        'first_filters': {'type': 'int', 'min': 20, 'max': 100},
                        'second_filters': {'type': 'int', 'min': 20, 'max': 50},
                        'third_filters': {'type': 'int', 'min': 10, 'max': 20}}

default_hyperparams={'gamma': 2.0,
                     'alpha': 0.7,
                     'first_filters': 30,
                     'second_filters': 20,
                     'third_filters': 10}

validate_every = 1
save_every = 1

train_dataset = Embedded_k562_ATAC_train_dataset(data_path, transform=transformer)
valid_dataset = Embedded_k562_ATAC_validation_dataset(data_path, transform=transformer)
data_cast = lambda x: torch.autograd.Variable(x).float()
label_cast = lambda y: torch.autograd.Variable(y).float()


class BindSpaceNet(nn.Module):

    def __init__(self, input_size=(1, 281, 300), num_factors=24, hyperparams_dict=default_hyperparams):

        super(BindSpaceNet, self).__init__()
        self.relu = nn.SELU()
        first_filters = hyperparams_dict['first_filters']
        second_filters = hyperparams_dict['second_filters']
        third_filters = hyperparams_dict['third_filters']
        
        self.conv1 = nn.utils.weight_norm(nn.Conv2d(1, first_filters, (1,300)))
        self.pool1 = nn.MaxPool2d(kernel_size=(3,1)) 

        # Here is where I should think about what makes sense as a region to
        # pool over: how much effetive sequence space do I want to consider?
        # kernel_size(1,3) gives me effectively 23 bases of consideration
        # Can also try Lp pooling for large P

        self.conv2 = nn.utils.weight_norm(nn.Conv2d(first_filters,second_filters,(30,1)))
        
        self.conv3 = nn.utils.weight_norm(nn.Conv2d(second_filters, third_filters, 
                                                   (20,1)))
        
        conv_size = self._get_conv_output(input_size)


        self.fc1 = nn.utils.weight_norm(nn.Linear(conv_size, num_factors))


    def forward(self, input):

        x = self.pool1(self.relu(self.conv1(input)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # flatten layer
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))

        return x

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
        x_c3 = self.conv3(x_c2)
        return x_c3

def reinitialize_model(num_factors=19, hyperparams_dict=default_hyperparams):
    net = BindSpaceNet(num_factors=num_factors, hyperparams_dict=hyperparams_dict)
    net.apply(tmu.init_weights)
    return net

def get_additional_losses(net, hyperparams_dict):
    ''' Return a list of additional terms for the loss function '''
    return []

def get_train_loss(hyperparams_dict=default_hyperparams):
    ''' Set up the focal loss for training '''
    return tmu.FocalLoss(gamma=hyperparams_dict['gamma'], alpha=hyperparams_dict['alpha'])
    
train_loss = get_train_loss()
valid_loss = nn.BCEWithLogitsLoss(size_average=False)

net = BindSpaceNet(num_factors=num_factors, hyperparams_dict=default_hyperparams)
net.apply(tmu.init_weights)

# Collect weight, bias parameters for regularization
weights, biases, sparse_weights = tmu.get_model_param_lists(net)

# Initialize the params, put together the arguments for the optimizer
additional_losses = get_additional_losses(net, default_hyperparams)
optimizer, optimizer_param_dicts = tmu.initialize_optimizer(weights, biases, 
                                                            sparse_weights, 
    default_hyperparams)

optimizer_kwargs = {'lr': initial_lr}

learning_rate_scheduler = torch.optim.lr_scheduler.StepLR
lrs_kwargs = {'step_size': 5, 'gamma': 0.1}

