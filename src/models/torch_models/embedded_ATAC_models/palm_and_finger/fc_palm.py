import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from load_pytorch import BindspaceProbeDataset
from load_pytorch import ProbeReshapeTransformer
from utils import torch_model_construction_utils as tmu

data_path = os.path.expanduser("~/projects/SeqDemote/data/ATAC/K562/K562_embed_TV_annotated_split.h5")
save_dir = "BindSpace_embedding_extension"

num_factors = 19  
batch_size = 64
momentum = None
embedded_seq_len = 84300
embedding_dim_len = 300
cuda = True
initial_lr = 0.005
max_batches_per_epoch = 500000

model_hyperparams_dict={'first_layer': {'type': 'int', 'min': 20, 'max': 100},
                        'second_layer': {'type': 'int', 'min': 20, 'max': 50},
                        'dropout': {'type': 'float', 'min': 0.1, 'max': 0.5},
                        'weight_lambda': {'type': 'float', 'min': 1e-5, 'max': 1e-1},
                        'bias_lambda': {'type': 'float', 'min': 1e-5, 'max': 1e-1},
                        'sparse_lambda': {'type': 'float', 'min': 1e-5, 'max': 1e-1}}


default_hyperparams={'first_layer': 100,
                     'second_layer': 20,
                     'dropout': 0.2,
                     'weight_lambda': 5e-2,
                     'bias_lambda': 5e-2,
                     'sparse_lambda': 10e-2}

validate_every = 1
save_every = 1

train_loss = nn.BCEWithLogitsLoss(size_average=False)
valid_loss = nn.BCEWithLogitsLoss(size_average=False)
train_dataset = BindspaceProbeDataset(data_path, dataset='training', transform=None)
valid_dataset = BindspaceProbeDataset(data_path, dataset='validation', transform=None)
data_cast = lambda x: torch.autograd.Variable(x).float()
label_cast = lambda y: torch.autograd.Variable(y).float()


class BindSpaceNet(nn.Module):

    def __init__(self, input_size=300, num_factors=19, hyperparams_dict=default_hyperparams):

        super(BindSpaceNet, self).__init__()
        self.relu = nn.SELU()
        fc1_out_features = hyperparams_dict['first_layer']
        fc2_out_features = hyperparams_dict['second_layer']
        self.dropout = nn.Dropout(p=hyperparams_dict['dropout'])
        
        # shared parameters
        self.fc1 = nn.utils.weight_norm(nn.Linear(input_size, fc1_out_features))
        
        # factor specific 'finger' parameters
        self.fingers = nn.ModuleList([nn.Sequential(nn.utils.weight_norm(nn.Linear(fc1_out_features,fc2_out_features)),
                                                    self.relu, self.dropout,
                                                    nn.utils.weight_norm(nn.Linear(fc2_out_features, 1))) for f in range(num_factors)])

    def forward(self, input):

        # shared forward computation
        x = self.relu(self.fc1(input))
            
        # factor-specific forward computations
        return [finger(x) for finger in self.fingers]

def reinitialize_model(num_factors=19, hyperparams_dict=default_hyperparams):
    net = BindSpaceNet(num_factors=num_factors, hyperparams_dict=hyperparams_dict)
    net.apply(tmu.init_weights)
    return net

def get_additional_losses(net, hyperparams_dict):
    ''' Return a list of additional terms for the loss function '''
    return []

net = BindSpaceNet(num_factors=num_factors, hyperparams_dict=default_hyperparams)
# Initialize model params
net.apply(tmu.init_weights)

# Collect weight, bias parameters for regularization
weights, biases, sparse_weights = tmu.get_model_param_lists(net)

# Initialize the params, put together the arguments for the optimizer
additional_losses = get_additional_losses(net, default_hyperparams)
optimizer, optimizer_param_dicts = tmu.initialize_optimizer(weights, biases, 
                                                            sparse_weights, 
    default_hyperparams)

optimizer_kwargs = {'lr': initial_lr}