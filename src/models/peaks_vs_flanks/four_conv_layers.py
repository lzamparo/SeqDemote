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
validate_every = 1
save_every = 5
data_loader = load.HematopoeticDataLoader(chunk_size=chunk_size, batch_size=batch_size, num_chunks_train=num_chunks_train) 


# Refs to lasagne conv layers
Conv2DLayer = nn.layers.Conv2DLayer
MaxPool2DLayer = nn.layers.MaxPool2DLayer

BatchNormLayer = nn.layers.BatchNormLayer

### TODO: Try to evade pooling for a few rounds, doing so only at the end, and try not using so many units in dense layer to cut down on params.
### at least until data set gets larger.
def build_model():

    l0 = nn.layers.InputLayer((batch_size, data_rows, 1, data_cols))  ## TODO: first dim maybe be chunk_size
    l1a = Conv2DLayer(l0, num_filters=200, filter_size=(1, 3), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l1b = BatchNormLayer(l1a)
    l1c = nn.layers.NonlinearityLayer(l1b,nonlinearity=nn.nonlinearities.leaky_rectify)

    l2a = Conv2DLayer(l1c, num_filters=100, filter_size=(1, 6), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l2b = BatchNormLayer(l2a)
    l2c = nn.layers.NonlinearityLayer(l2b,nonlinearity=nn.nonlinearities.leaky_rectify)
    l2d = MaxPool2DLayer(l2c, pool_size=(1, 4), stride=(1, 2))

    l3a = Conv2DLayer(l2d, num_filters=100, filter_size=(1, 9), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l3b = BatchNormLayer(l3a)
    l3c = nn.layers.NonlinearityLayer(l3b,nonlinearity=nn.nonlinearities.leaky_rectify)
    l3d = MaxPool2DLayer(l3c, pool_size=(1, 4), stride=(1, 2))
    
    l4a = Conv2DLayer(l3d, num_filters=50, filter_size=(1, 12), W=nn.init.Orthogonal(gain='relu'), b=nn.init.Constant(0.1), nonlinearity=None, untie_biases=True)
    l4b = BatchNormLayer(l4a)
    l4c = nn.layers.NonlinearityLayer(l4b,nonlinearity=nn.nonlinearities.leaky_rectify)
    l4d = MaxPool2DLayer(l4c, pool_size=(1, 4), stride=(1, 2))
    
    ### output dims of l3d should be (n_batches, 100, 10)
    #l4a = nn.layers.ReshapeLayer(l3d, shape=(batch_size,2000)) ## produces the same output shape, and without the need to specify the shape.
    l4a = nn.layers.FlattenLayer(l4d)
    l4b = nn.layers.DenseLayer(l4a, 1000)
    l4c = BatchNormLayer(l4b)
    l4d = nn.layers.DropoutLayer(l4c, p=0.5)

    
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