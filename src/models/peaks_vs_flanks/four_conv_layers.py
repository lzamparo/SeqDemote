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
model_params_dict = {'l1filters': {'type': 'int', 'min': 100, 'max': 300},
                     'l1filter_size': {'type': 'int', 'min': 3, 'max': 6}, 
                     'l1pool_size': {'type': 'int', 'min': 2, 'max': 4}, 
               'l1stride': {'type': 'int', 'min': 2, 'max': 4},
               'l2filters': {'type': 'int', 'min': 80, 'max': 200},
               'l2filter_size': {'type': 'int', 'min': 6, 'max': 9}, 
               'l2pool_size': {'type': 'int', 'min': 2, 'max': 4}, 
               'l2stride': {'type': 'int', 'min': 2, 'max': 4},
               'l3filters': {'type': 'int', 'min': 80, 'max': 200},
               'l3filter_size': {'type': 'int', 'min': 9, 'max': 12}, 
               'l3pool_size': {'type': 'int', 'min': 2, 'max': 4}, 
               'l3stride': {'type': 'int', 'min': 2, 'max': 4},
               'l4filters': {'type': 'int', 'min': 60, 'max': 100},
               'l4filter_size': {'type': 'int', 'min': 12, 'max': 18}, 
               'l4pool_size': {'type': 'int', 'min': 2, 'max': 4}, 
               'l4stride': {'type': 'int', 'min': 2, 'max': 4},               
               'l5dense_size': {'type': 'int', 'min': 200, 'max': 400}, 
               'l5dropout': {'type': 'float', 'min': 0.2, 'max': 0.6}}

num_epochs = 8
validate_every = 2
save_every = 2
data_loader = load.HematopoeticDataLoader(chunk_size=chunk_size, batch_size=batch_size, num_chunks_train=num_chunks_train) 


# Refs to lasagne conv layers
Conv2DLayer = nn.layers.Conv2DLayer
MaxPool2DLayer = nn.layers.MaxPool2DLayer
BatchNormLayer = nn.layers.BatchNormLayer

### TODO: Try to evade pooling for a few rounds, doing so only at the end, and try not using so many units in dense layer to cut down on params.
### at least until data set gets larger.
def build_model(params_dict=None):
    ''' Construct the network from the params dict (or the established default dict below if none is provided) '''
    if not params_dict:
        params_dict = {'l1filters': 200, 'l1filter_size': 3,
                       'l2filters': 100, 'l2filter_size': 6, 'l2pool_size': 4, 'l2stride': 2,
                       'l3filters': 100, 'l3filter_size': 9, 'l3pool_size': 4, 'l3stride': 2,
                       'l4filters': 50, 'l4filter_size': 12, 'l4pool_size': 4, 'l4stride': 2,
                       'l5dense_size': 1000, 'l5dropout': 0.5}    
    

    l0 = nn.layers.InputLayer((batch_size, data_rows, 1, data_cols))  ## TODO: first dim maybe be chunk_size
    l1a = Conv2DLayer(l0, num_filters=params_dict['l1filters'], filter_size=(1, params_dict['l1filter_size']), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l1b = BatchNormLayer(l1a)
    l1c = nn.layers.NonlinearityLayer(l1b,nonlinearity=nn.nonlinearities.leaky_rectify)

    l2a = Conv2DLayer(l1c, num_filters=params_dict['l2filters'], filter_size=(1, params_dict['l2filter_size']), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l2b = BatchNormLayer(l2a)
    l2c = nn.layers.NonlinearityLayer(l2b,nonlinearity=nn.nonlinearities.leaky_rectify)
    l2d = MaxPool2DLayer(l2c, pool_size=(1, params_dict['l2pool_size']), stride=(1, params_dict['l2stride']))

    l3a = Conv2DLayer(l2d, num_filters=params_dict['l3filters'], filter_size=(1, params_dict['l3filter_size']), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l3b = BatchNormLayer(l3a)
    l3c = nn.layers.NonlinearityLayer(l3b,nonlinearity=nn.nonlinearities.leaky_rectify)
    l3d = MaxPool2DLayer(l3c, pool_size=(1, params_dict['l3pool_size']), stride=(1, params_dict['l3stride']))
    
    l4a = Conv2DLayer(l3d, num_filters=params_dict['l4filters'], filter_size=(1, params_dict['l4filter_size']), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l4b = BatchNormLayer(l4a)
    l4c = nn.layers.NonlinearityLayer(l4b,nonlinearity=nn.nonlinearities.leaky_rectify)
    l4d = MaxPool2DLayer(l4c, pool_size=(1, params_dict['l4pool_size']), stride=(1, params_dict['l4stride']))
    
    ### output dims of l3d should be (n_batches, 100, 10)
    #l4a = nn.layers.ReshapeLayer(l3d, shape=(batch_size,2000)) ## produces the same output shape, and without the need to specify the shape.
    l5a = nn.layers.FlattenLayer(l4d)
    l5b = nn.layers.DenseLayer(l4a, params_dict['l5dense_size'])
    l5c = BatchNormLayer(l4b)
    l5d = nn.layers.DropoutLayer(l4c, p=params_dict['l5dropout'])

    
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
