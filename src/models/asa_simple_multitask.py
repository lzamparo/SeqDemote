import os
import numpy as np
from collections import OrderedDict
from utils.train_utils import log_lr_schedule, apply_aggregated_threevec_momentum


import theano 
import theano.tensor as T
import lasagne as nn

import generators
import load 

data_rows = 4 # probably don't need this param specified here
data_cols = 300 # probably don't need this param specified here

train_in_path ='/data/train_in'
train_out_path = '/labels/train_out'
valid_in_path = '/data/valid_in'
valid_out_path = '/labels/valid_out'
test_in_path = '/data/test_in'
test_out_path = '/labels/test_out'

data_path = os.path.expanduser("~/projects/SeqDemote/data/ATAC/mouse_asa/mouse_asa.h5")

### training params
batch_size = 128
chunk_size = 199078
num_chunks_train = 1
num_chunks_valid = 1
momentum = 0.98
weight_norm = 7  ### called after each parameter update, during training, use lasagne.updates.norm_constraint()
#resume_path = '/cbio/cllab/home/zamparol/projects/SeqDemote/src/models/checkpoints/basset_onehot.py-gpu-2-5.local-20160629-185410.pkl'

# set schedule for learning rate decreases
learning_rate_schedule = {
    0: 0.01,
    10: 0.001,
    20: 0.0001,
}
validate_every = 1
save_every = 5
data_loader = load.StandardDataLoader(chunk_size=chunk_size, 
                                      batch_size=batch_size, 
                                      num_chunks_train=num_chunks_train,
                                      train_in_path=train_in_path,
                                      train_out_path=train_out_path,
                                      test_in_path=test_in_path,
                                      test_out_path=test_out_path,
                                      valid_in_path=valid_in_path,
                                      valid_out_path=valid_out_path,
                                      data_path=data_path) 

# Refs to lasagne conv layers
Conv2DLayer = nn.layers.Conv2DLayer
MaxPool2DLayer = nn.layers.MaxPool2DLayer
BatchNormLayer = nn.layers.BatchNormLayer


# This is the DeepLIFT simulation arch for testing
#Convolution with 50 filters of width 11
#• ReLU
#• Convolution with 50 filters of width 11
#• ReLU
#• Global Average Pooling
#• Fully connected layer with 50 neurons
#• ReLU
#• Dropout layer with probability 0.5 of leaving out units
#• Fully connected layer with 3 neurons
#• Sigmoid output


def build_model():

    l0 = nn.layers.InputLayer((batch_size, data_rows, 1, data_cols))  ## TODO: first dim maybe be chunk_size
    
    l1a = Conv2DLayer(l0, num_filters=50, filter_size=(1, 11), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l1b = BatchNormLayer(l1a)
    l1c = nn.layers.NonlinearityLayer(l1b)
    
    l2a = Conv2DLayer(l1c, num_filters=50, filter_size=(1, 11), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l2b = BatchNormLayer(l2a)
    l2c = nn.layers.NonlinearityLayer(l2b)

    l3 = nn.layers.GlobalPoolLayer(l2c)  #Think this leads to a broadcasting bug:
    # https://github.com/Theano/Theano/issues/5384, https://github.com/Theano/Theano/issues/5384#issuecomment-273493711
    #l3 = nn.layers.FeaturePoolLayer(l2c, 50,1,pool_function=T.mean)
    l3b = nn.layers.FlattenLayer(l3)
    
    l4a = nn.layers.DenseLayer(l3b, 50)
    l4b = nn.layers.BatchNormLayer(l4a)
    l4c = nn.layers.DropoutLayer(l4b, p=0.5)

    l5 = nn.layers.DenseLayer(l4c, num_units=5, nonlinearity=nn.nonlinearities.linear)

    return l0, l5


def poisson_nll(y, t, log_input=True, reduce_avg=True, eps=1e-15):
    """
    Produce an expression to compute the Poisson negative log-likelihood loss
    
    log_input: Bool, compute loss or exp(loss)
    reduce_avg: Bool, return the aggretage(loss)
    eps: np.float, added to y so as to avoid underflow for log(y)
    
    Inspired by Pytorch implementation:
    
    if log_input:
        loss = torch.exp(input) - target * input
    else:
        loss = input - target * torch.log(input + eps)
    if not reduce:
        return loss
    if size_average:
        return torch.mean(loss)
    return torch.sum(loss)
    """
    
    if log_input:
        loss = T.exp(y) - (t * y)
    else:
        loss = y - t*(T.log(y + eps))
    if reduce_avg:
        return nn.objectives.aggregate(loss)
    else:
        return loss
    

def build_objective(l_ins, l_out, targets, deterministic=False):
    lambda_reg = 0.005
    params = nn.layers.get_all_params(l_out, regularizable=True)
    reg_term = nn.regularization.regularize_layer_params(l_out, nn.regularization.l2, tags={'regularizable': True})
    prediction = nn.layers.get_output(l_out, deterministic=deterministic)
    loss = poisson_nll(prediction, targets, reduce_avg=False) + lambda_reg * reg_term 
    return nn.objectives.aggregate(loss)


def build_updates(train_loss, all_params, learning_rate, momentum):   
    updates = nn.updates.rmsprop(train_loss, all_params, learning_rate, momentum)
    #normed_updates = OrderedDict((param, nn.updates.norm_constraint(updates[param], weight_norm)) if param.ndim > 1 else (param, updates[param]) for param in updates)  
    return updates
