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
# hyper-parameter range dictionary for training with spearmint
model_params_dict = {'l1filters': {'type': 'int', 'min': 10, 'max': 30},
                     'l1filter_size': {'type': 'int', 'min': 5, 'max': 10}, 
                     'l1pool_size': {'type': 'int', 'min': 2, 'max': 4}, 
               'l1stride': {'type': 'int', 'min': 1, 'max': 2},
               'l2filters': {'type': 'int', 'min': 10, 'max': 30},
               'l2filter_size': {'type': 'int', 'min': 10, 'max': 15}, 
               'l2pool_size': {'type': 'int', 'min': 2, 'max': 4}, 
               'l2stride': {'type': 'int', 'min': 1, 'max': 2},
               'l3filters': {'type': 'int', 'min': 10, 'max': 30},
               'l3filter_size': {'type': 'int', 'min': 10, 'max': 15}, 
               'l3pool_size': {'type': 'int', 'min': 2, 'max': 4}, 
               'l3stride': {'type': 'int', 'min': 1, 'max': 2},               
               'l4dense_size': {'type': 'int', 'min': 100, 'max': 200}, 
               'l4dropout': {'type': 'float', 'min': 0.2, 'max': 0.6}}

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
        params_dict = {'l1filters': 20, 'l1filter_size': 10, 'l1pool_size': 2, 'l1stride': 1,
                       'l2filters': 10, 'l2filter_size': 10, 'l2pool_size': 2, 'l2stride': 1,
                       'l3filters': 10, 'l3filter_size': 10, 'l3pool_size': 2, 'l3stride': 1,
                       'l4dense_size': 100, 'l4dropout': 0.5}        

    l0 = nn.layers.InputLayer((batch_size, data_rows, 1, data_cols))  ## TODO: first dim maybe be chunk_size
    l1a = Conv2DLayer(l0, num_filters=params_dict['l1filters'], filter_size=(1, params_dict['l1filter_size']), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l1b = BatchNormLayer(l1a)
    l1c = nn.layers.NonlinearityLayer(l1b,nonlinearity=nn.nonlinearities.leaky_rectify)
    l1d = MaxPool2DLayer(l1c, pool_size=(1, params_dict['l1pool_size']), stride=(1, params_dict['l1stride']))

    l2a = Conv2DLayer(l1d, num_filters=params_dict['l2filters'], filter_size=(1, params_dict['l2filter_size']), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l2b = BatchNormLayer(l2a)
    l2c = nn.layers.NonlinearityLayer(l2b,nonlinearity=nn.nonlinearities.leaky_rectify)
    l2d = MaxPool2DLayer(l2c, pool_size=(1, params_dict['l2pool_size']), stride=(1, params_dict['l2stride']))

    l3a = Conv2DLayer(l2d, num_filters=params_dict['l3filters'], filter_size=(1, params_dict['l3filter_size']), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l3b = BatchNormLayer(l3a)
    l3c = nn.layers.NonlinearityLayer(l3b,nonlinearity=nn.nonlinearities.leaky_rectify)
    l3d = MaxPool2DLayer(l3c, pool_size=(1, params_dict['l3pool_size']), stride=(1, params_dict['l3stride']))
    
    ### output dims of l3d should be (n_batches, 100, 10)
    l4a = nn.layers.FlattenLayer(l3d)
    l4b = nn.layers.DenseLayer(l4a, params_dict['l4dense_size'])
    l4c = BatchNormLayer(l4b)
    l4d = nn.layers.DropoutLayer(l4c, p=params_dict['l4dropout'])
    
    l5 = nn.layers.DenseLayer(l4d, num_units=1, nonlinearity=nn.nonlinearities.sigmoid)

    return l0, l5


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
