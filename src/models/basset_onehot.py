import numpy as np

import theano 
import theano.tensor as T

import lasagne as nn

import data
import load 

data_rows = 4
data_cols = 600

batch_size = 128
chunk_size = 4096
num_chunks_train = 1880000 // 4096

# TODO: update these hyper-params to reflect Basset settings
momentum = 0.9
learning_rate_schedule = {
    0: 0.003,
    700: 0.0003,
    800: 0.00003,
}

validate_every = 20
save_every = 20

data_loader = load.DNaseDataLoader() # do I need to specify chunks, chunk size here?

#data_loader = load.ZmuvRescaledDataLoader(estimate_scale=estimate_scale, num_chunks_train=num_chunks_train,
    #patch_size=patch_size, chunk_size=chunk_size, augmentation_params=augmentation_params,
    #augmentation_transforms_test=augmentation_transforms_test, validation_split_path=validation_split_path)

### The output of the basset model with fewer filters
#(1): nn.SpatialConvolution(4 -> 150, 19x1) ** should be 300
    #(2): nn.SpatialBatchNormalization
    #(3): nn.ReLU
    #(4): nn.SpatialMaxPooling(3,1,3,1)
    #(5): nn.SpatialConvolution(150 -> 100, 11x1) ** should be 200
    #(6): nn.SpatialBatchNormalization
    #(7): nn.ReLU
    #(8): nn.SpatialMaxPooling(4,1,4,1)
    #(9): nn.SpatialConvolution(100 -> 100, 7x1) ** should be 200
    #(10): nn.SpatialBatchNormalization
    #(11): nn.ReLU
    #(12): nn.SpatialMaxPooling(4,1,4,1)
    #(13): nn.Reshape(1000)
    #(14): nn.Linear(1000 -> 1000)
    #(15): nn.BatchNormalization
    #(16): nn.ReLU
    #(17): nn.Dropout(0.300000)
    #(18): nn.Linear(1000 -> 1000)
    #(19): nn.BatchNormalization
    #(20): nn.ReLU
    #(21): nn.Dropout(0.300000)
    #(22): nn.Linear(1000 -> 164)
    #(23): nn.Sigmoid
    
    
### The pretrained_params.txt file
    #conv_filters    300
    #conv_filters    200
    #conv_filters    200
    #conv_filter_sizes       19
    #conv_filter_sizes       11
    #conv_filter_sizes       7
    #pool_width              3
    #pool_width              4
    #pool_width              4
    #hidden_units            1000
    #hidden_units            1000
    #hidden_dropouts         0.3
    #hidden_dropouts         0.3
    #learning_rate           0.002
    #weight_norm             7      *** called after each parameter update, during training.  See https://github.com/nicholas-leonard/dpnn#nn.Module.maxParamNorm for details, not sure if supported in Lasagne.
    #momentum                0.98    
    
    # shortcut to Lasagne layers needed
    
Conv2DLayer = nn.layers.dnn.Conv2DDNNLayer
MaxPool2DLayer = nn.layers.dnn.MaxPool2DDNNLayer
BatchNormLayer = nn.layers.BatchNormLayer

def build_model():
    # TODO: update for basset model via output script.  Output layer should be dense layer of 164, with softmax nonlinearity (I think, check this).
    l0 = nn.layers.InputLayer((batch_size, 1, data_rows, data_cols))

    l1 = Conv2DLayer(l0c, num_filters=32, filter_size=(19, 1), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l1b = Conv2DLayer(l1a, num_filters=16, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l1 = MaxPool2DLayer(l1b, ds=(3, 3), strides=(2, 2))
    l1r = dihedral_fast.CyclicConvRollLayer(l1)

    l2a = Conv2DLayer(l1r, num_filters=64, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l2b = Conv2DLayer(l2a, num_filters=32, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l2 = MaxPool2DLayer(l2b, ds=(3, 3), strides=(2, 2))
    l2r = dihedral_fast.CyclicConvRollLayer(l2)

    l3a = Conv2DLayer(l2r, num_filters=128, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l3b = Conv2DLayer(l3a, num_filters=128, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l3c = Conv2DLayer(l3b, num_filters=64, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l3 = MaxPool2DLayer(l3c, ds=(3, 3), strides=(2, 2))
    l3r = dihedral_fast.CyclicConvRollLayer(l3)

    l4a = Conv2DLayer(l3r, num_filters=256, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l4b = Conv2DLayer(l4a, num_filters=256, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)
    l4c = Conv2DLayer(l4b, num_filters=128, filter_size=(3, 3), border_mode="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu, untie_biases=True)    
    l4 = MaxPool2DLayer(l4c, ds=(3, 3), strides=(2, 2))
    l4r = dihedral_fast.CyclicConvRollLayer(l4)
    l4f = nn.layers.flatten(l4r)

    l5 = nn.layers.DenseLayer(nn.layers.dropout(l4f, p=0.5), num_units=1024, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=None)
    l5fp = nn.layers.FeaturePoolLayer(l5, ds=2)
    l5r = dihedral_fast.CyclicRollLayer(l5fp)

    l6 = nn.layers.DenseLayer(nn.layers.dropout(l5r, p=0.5), num_units=1024, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=None)
    l6fp = nn.layers.FeaturePoolLayer(l6, ds=2)
    l6m = dihedral.CyclicPoolLayer(l6fp, pool_function=nn_plankton.rms)

    l7 = nn.layers.DenseLayer(nn.layers.dropout(l6m, p=0.5), num_units=data.num_classes, nonlinearity=T.nnet.softmax, W=nn_plankton.Orthogonal(1.0))

    return [l0], l7



def build_objective(l_ins, l_out):
    # TODO: update for basset loss, regularization of params
    lambda_reg = 0.0005
    params = nn.layers.get_all_non_bias_params(l_out)
    reg_term = sum(T.sum(p**2) for p in params)

    def loss(y, t):
        return nn_plankton.log_loss(y, t) + lambda_reg * reg_term

    return nn.objectives.Objective(l_out, loss_function=loss)
