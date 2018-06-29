import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from load_pytorch import Embedded_k562_ATAC_train_dataset, Embedded_k562_ATAC_validation_dataset
from load_pytorch import EmbeddingReshapeTransformer

data_path = os.path.expanduser("~/projects/SeqDemote/data/ATAC/K562/K562_embed_TV_split.h5")
save_dir = "BindSpace_embedding_extension"

num_factors = 19  
batch_size = 32
momentum = None
embedded_seq_len = 84300
embedding_dim_len = 300
transformer = EmbeddingReshapeTransformer(embedding_dim_len, embedded_seq_len)
cuda = True

learning_rate_schedule = {
0: 0.005,
10: 0.002,
20: 0.0001}

model_hyperparams_dict={'orth_lambda': {'type': 'float', 'min': 1e-6, 'max': 1.0},
                        'weight_lambda': {'type': 'float', 'min': 1e-8, 'max': 1e-1},
                        'bias_lambda': {'type': 'float', 'min': 1e-8, 'max': 1e-1},
                        'sparse_lambda': {'type': 'float', 'min': 1e-8, 'max': 1e-1}}

default_hyperparams={'orth_lambda': 1e-6,
                     'weight_lambda': 5e-3,
                     'bias_lambda': 5e-3,
                     'sparse_lambda': 10e-3}

validate_every = 1
save_every = 1

train_loss = nn.BCEWithLogitsLoss(size_average=False)
valid_loss = nn.BCEWithLogitsLoss(size_average=False)
train_dataset = Embedded_k562_ATAC_train_dataset(data_path, transform=transformer)
valid_dataset = Embedded_k562_ATAC_validation_dataset(data_path, transform=transformer)
data_cast = lambda x: torch.autograd.Variable(x).float()
label_cast = lambda y: torch.autograd.Variable(y).float()


class BindSpaceNet(nn.Module):
    
    def __init__(self, input_size=(1, 281, 300), num_factors=19):
        
        super(BindSpaceNet, self).__init__()
        self.relu = nn.SELU()
        
        self.orth_conv1 = nn.utils.weight_norm(nn.Conv2d(1, 20, (1,300)))
        self.pool1 = nn.MaxPool2d(kernel_size=(3,1)) 
        
        # Here is where I should think about what makes sense as a region to
        # pool over: how much effetive sequence space do I want to consider?
        # kernel_size(1,3) gives me effectively 23 bases of consideration
        # Can also try Lp pooling for large P

        conv_size = self._get_conv_output(input_size)

        # I'm not sure I want to reshape the output of all conv-pool
        # filters into one long vector; think it makes sense to do 
        # something else; 
        # - groups of locally connected layers to further
        # shrink the output before combining them?
        # - encourage a learned hierarchy of groups?
         
        self.sparse_fc1 = nn.utils.weight_norm(nn.Linear(conv_size, num_factors))
        
        
    def forward(self, input):
        
        x = self.pool1(self.relu(self.orth_conv1(input)))
        
        # flatten layer
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.sparse_fc1(x))
        
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
        x_c1 = self.orth_conv1(x)
        x_p1 = self.pool1(x_c1)

        return x_p1



def init_weights(m, gain=nn.init.calculate_gain('relu')):
    ''' Recursively initalizes the weights of a network. '''
    
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    
    if isinstance(m, nn.Conv2d):
        torch.nn.init.orthogonal_(m.weight, gain)
        m.bias.data.fill_(0.1)
    

### Functions that allow for re-initialization of model and
### optimizer to tune hyperparameters 

def reinitialize_model(num_factors=19):
    net = BindSpaceNet(num_factors=num_factors)
    net.apply(init_weights)
    return net


def get_model_param_lists(net):
    biases, weights, sparse_weights = [], [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            biases += [p]
            
        elif 'sparse' in name:
            sparse_weights += [p]
        
        else:
            weights += [p]  
            
    return biases, weights, sparse_weights
    
def orthogonal_filter_penalty(net, orth_lambda=1e-6):
    ''' Impose an additional decorrelative penalty on the conv filters '''
    
    for name, p in net.named_parameters():
        if 'orth' in name and 'weight_v' in name:
            p_flattened = p.view(p.size(0),-1)
            WWt = torch.mm(p_flattened, torch.transpose(p_flattened,0,1))
            WWt -= torch.Tensor(torch.eye(p_flattened.size(0)))
            orth_loss = orth_lambda * WWt.sum()
    return orth_loss

def get_additional_losses(net, hyperparams_dict):
    return [orthogonal_filter_penalty(net, hyperparams_dict['orth_lambda'])]

def initialize_optimizer(weights, biases, sparse_weights, hyperparams_dict):
    ''' Initialize the params, put together the arguments for the optimizer '''

    weight_lambda = hyperparams_dict['weight_lambda']
    bias_lambda = hyperparams_dict['bias_lambda']
    sparse_lambda = hyperparams_dict['sparse_lambda']
    
    optimizer = torch.optim.Adam
    optimizer_param_dicts = [
        {'params': weights, 'weight_decay': weight_lambda},
            {'params': biases, 'weight_decay': bias_lambda},
            {'params': sparse_weights, 'weight_decay': sparse_lambda}            
    ]
    return optimizer, optimizer_param_dicts
 
net = BindSpaceNet(num_factors=num_factors)
net.apply(init_weights)

# Collect weight, bias parameters for regularization
weights, biases, sparse_weights = get_model_param_lists(net) 

# Collect optimizer, additional losses     
additional_losses = get_additional_losses(net, default_hyperparams)
optimizer, optimizer_param_dicts = initialize_optimizer(weights, biases, 
                                                       sparse_weights,default_hyperparams) 
optimizer_kwargs = {'lr': learning_rate_schedule[0]}
       
