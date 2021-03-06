import numpy as np
from collections import OrderedDict
from utils.train_utils import log_lr_schedule

import theano 
import theano.tensor as T

import lasagne as nn

import generators
import load 

data_rows = 4 # probably don't need this param specified here
data_cols = 600 # probably don't need this param specified here

### training params
task_type = 'peaks_vs_flanks'
save_dir = 'peaks_vs_flanks'

batch_size = 128
chunk_size = 4096
num_chunks_train = 240943 // chunk_size
num_chunks_valid = 51528 // chunk_size
momentum = 0.98
weight_norm = 7  ### called after each parameter update, during training, use lasagne.updates.norm_constraint()
#resume_path = '/cbio/cllab/home/zamparol/projects/SeqDemote/src/models/checkpoints/basset_onehot.py-gpu-2-5.local-20160629-185410.pkl'

# set schedule for learning rate decreases
learning_rate_schedule = {
    0: 0.003,
    700: 0.0003,
    800: 0.00003,
}
model_params_dict = {'l1filters': {'type': 'int', 'min': 50, 'max': 300},
                     'l1filter_size': {'type': 'int', 'min': 3, 'max': 30}, 
               'l1pool_size': {'type': 'int', 'min': 3, 'max': 5}, 
               'l1stride': {'type': 'int', 'min': 2, 'max': 4},
               'l2dense_size': {'type': 'int', 'min': 50, 'max': 100}, 
               'l2dropout': {'type': 'float', 'min': 0.1, 'max': 0.6}}

num_epochs = 8
validate_every = 2
save_every = 2

data_loader = load.HematopoeticDataLoader(chunk_size=chunk_size, batch_size=batch_size, num_chunks_train=num_chunks_train) 


# Refs to lasagne conv layers
Conv2DLayer = nn.layers.Conv2DLayer
MaxPool2DLayer = nn.layers.MaxPool2DLayer
BatchNormLayer = nn.layers.BatchNormLayer

def build_model(params_dict=None):
    ''' Construct the network from the params dict (or the established default dict below if none is provided) '''
    if not params_dict:
        params_dict = {'l1filters': 100, 'l1filter_size': 10, 'l1pool_size': 4, 'l1stride': 2,
                       'l2dense_size': 200, 'l2dropout': 0.5}   
        
    l0 = nn.layers.InputLayer((batch_size, data_rows, 1, data_cols))  ## TODO: first dim maybe be chunk_size
    l1a = Conv2DLayer(l0, num_filters=params_dict['l1filters'], filter_size=(1, params_dict['l1filter_size']), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l1b = BatchNormLayer(l1a)
    l1c = nn.layers.NonlinearityLayer(l1b,nonlinearity=nn.nonlinearities.leaky_rectify)
    l1d = MaxPool2DLayer(l1c, pool_size=(1, params_dict['l1pool_size']), stride=(1, params_dict['l1stride']))
    
    l2a = nn.layers.FlattenLayer(l1d)
    l2b = nn.layers.DenseLayer(l2a, params_dict['l2dense_size'])
    l2c = BatchNormLayer(l2b)
    l2d = nn.layers.DropoutLayer(l2c, p=params_dict['l2dropout'])
    
    l3 = nn.layers.DenseLayer(l2d, num_units=1, nonlinearity=nn.nonlinearities.sigmoid)

    return l0, l3


def build_objective(l_ins, l_out, targets, training_mode=True):
    lambda_reg = 0.0005
    params = nn.layers.get_all_params(l_out, regularizable=True)
    reg_term = nn.regularization.regularize_layer_params(l_out, nn.regularization.l2, tags={'regularizable': True})
    prediction = nn.layers.get_output(l_out, deterministic=training_mode)
    loss = nn.objectives.binary_crossentropy(prediction, targets) + lambda_reg * reg_term 
    return nn.objectives.aggregate(loss)


def build_updates(train_loss, all_params, learning_rate, momentum):   
    updates = nn.updates.adam(train_loss, all_params, learning_rate)
    #normed_updates = OrderedDict((param, nn.updates.norm_constraint(updates[param], weight_norm)) if param.ndim > 1 else (param, updates[param]) for param in updates)  
    return updates
