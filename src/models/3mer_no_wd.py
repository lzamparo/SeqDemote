import numpy as np
from collections import OrderedDict
from utils.train_utils import log_lr_schedule

import theano 
import theano.tensor as T

import lasagne as nn

import generators
import load 

data_rows = 64 # probably don't need this param specified here
data_cols = 598 # probably don't need this param specified here

### training params

batch_size = 128
chunk_size = 4096
num_chunks_train = 1880000 // chunk_size
num_chunks_valid = 70000 // chunk_size
momentum = 0.98
weight_norm = 7  ### called after each parameter update, during training, use lasagne.updates.norm_constraint()
dp = '/cbio/cllab/nobackup/zamparol/SeqDeep/data/encode_roadmap_3mer.h5'

# set schedule for learning rate decreases
base_lr = 0.002
learning_rate_schedule = log_lr_schedule(num_chunks_train, updates=4, base_lr)

validate_every = 1
save_every = 5
data_loader = load.StandardDataLoader(chunk_size=chunk_size, batch_size=batch_size, num_chunks_train=num_chunks_train, data_path=dp) 


# Refs to lasagne conv layers
Conv2DLayer = nn.layers.Conv2DLayer
MaxPool2DLayer = nn.layers.MaxPool2DLayer

BatchNormLayer = nn.layers.BatchNormLayer

def build_model():

    l0 = nn.layers.InputLayer((batch_size, data_rows, 1, data_cols))  ## TODO: first dim maybe be chunk_size
    l1a = Conv2DLayer(l0, num_filters=300, filter_size=(1, 19), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l1b = BatchNormLayer(l1a)
    l1c = nn.layers.NonlinearityLayer(l1b)
    l1d = MaxPool2DLayer(l1c, pool_size=(1, 3), stride=(1, 3))

    l2a = Conv2DLayer(l1d, num_filters=200, filter_size=(1, 11), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l2b = BatchNormLayer(l2a)
    l2c = nn.layers.NonlinearityLayer(l2b)
    l2d = MaxPool2DLayer(l2c, pool_size=(1, 4), stride=(1, 4))

    l3a = Conv2DLayer(l2d, num_filters=200, filter_size=(1, 7), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l3b = BatchNormLayer(l3a)
    l3c = nn.layers.NonlinearityLayer(l3b)
    l3d = MaxPool2DLayer(l3c, pool_size=(1, 4), stride=(1, 4))
    
    ### output dims of l3d should be (n_batches, 100, 10)
    #l4a = nn.layers.ReshapeLayer(l3d, shape=(batch_size,2000)) ## produces the same output shape, and without the need to specify the shape.
    l4a = nn.layers.FlattenLayer(l3d)
    l4b = nn.layers.DenseLayer(l4a, 1000)
    l4c = BatchNormLayer(l4b)
    l4d = nn.layers.DropoutLayer(l4c, p=0.3)
    
    l5a = nn.layers.DenseLayer(l4d, 1000)
    l5b = BatchNormLayer(l5a)
    l5c = nn.layers.DropoutLayer(l5b, p=0.3)
    
    l6 = nn.layers.DenseLayer(l5c, num_units=164, nonlinearity=nn.nonlinearities.sigmoid)

    return l0, l6


def build_objective(l_ins, l_out, targets, training_mode=False):
    prediction = nn.layers.get_output(l_out, deterministic=training_mode)
    loss = nn.objectives.binary_crossentropy(prediction, targets)
    return nn.objectives.aggregate(loss)


def build_updates(train_loss, all_params, learning_rate, momentum):   
    updates = nn.updates.rmsprop(train_loss, all_params, learning_rate, momentum)
    normed_updates = OrderedDict((param, nn.updates.norm_constraint(updates[param], weight_norm)) if param.ndim > 1 else (param, updates[param]) for param in updates)  
    return updates
