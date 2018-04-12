import sys
import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from load_pytorch import ATAC_Train_Dataset, ATAC_Valid_Dataset
from load_pytorch import SubsequenceTransformer
from utils import accounting, train_utils

import importlib.util
import pickle
from datetime import datetime, timedelta
import string

import matplotlib
matplotlib.use('agg')
import pylab as plt

from subprocess import Popen


if len(sys.argv) < 2:
    sys.exit("Usage: pytorch_train.py <configuration_name>")

model_config = sys.argv[1]
model_path_name = os.path.join(os.path.expanduser(os.getcwd()),'models',model_config)
spec = importlib.util.spec_from_file_location(model_config, model_path_name)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
expid = accounting.generate_expid(model_config)
expid = expid.split('/')[-1]

if hasattr(model_module, 'save_dir') and os.path.exists(os.path.join(train_utils.find_project_root(), 'results')):
    metadata_tmp_path = os.path.join(train_utils.find_project_root(), 'results', model_module.save_dir, expid + ".pkl")
else:
    metadata_tmp_path = os.path.join(train_utils.find_project_root(), 'results', expid + ".pkl")
    
print("Experiment ID: ", expid)

print("...Build model")
model = model_module.net

print("...number of parameters: ", train_utils.count_params(model.parameters()))
print("...layer output shapes:")
print(model)

print("...setting up loss functions ", flush=True)
if hasattr(model_module, "train_loss"):
    training_loss = model_module.train_loss
    valid_loss = model_module.valid_loss
else:
    training_loss = torch.nn.PoissonNLLLoss()
    valid_loss = torch.nn.PoissonNLLLoss()  # Need to figure out how to turn off Dropout in validation

print("...setting the learning rate schedule ")
if hasattr(model_module, 'learning_rate_schedule'):
    learning_rate_schedule = model_module.learning_rate_schedule
else:
    learning_rate_schedule = { 0: model_module.learning_rate }

print("...setting the optimization scheme ")
if hasattr(model_module, 'momentum'):
    momentum = model_module.momentum
    
else:
    momentum = 0.9
    
if hasattr(model_module, 'optimizer'):
    weights, biases = model_module.weights, model_module.biases
    optim = model_module.optimizer([
    {'params': weights, 'weight_decay': 1e-4},
                {'params': biases, 'weight_decay': 0}
                ], lr=learning_rate_schedule[0], momentum=momentum)
else:
    weights, biases = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            biases += [p]
        else:
            weights += [p]    
    optim = torch.optim.SGD([
    {'params': weights, 'weight_decay': 1e-4},
                {'params': biases, 'weight_decay': 0}
                ], lr=learning_rate_schedule[0], momentum=momentum)        
    
optim.zero_grad()

print("...setting up logging for losses ")
losses_train = []
losses_valid_log = []
losses_valid_auc = []
losses_valid_aupr = []
    
print("...Loading the data", flush=True)
if hasattr(model_module, 'batch_size'):
    batch_size = model_module.batch_size
else:
    batch_size = 128

if hasattr(model_module, 'train_dataset') and hasattr(model_module, 'valid_dataset'):
    train_dataset = model_module.train_dataset
    valid_dataset = model_module.valid_dataset    
else:
    print('Could not find specified datasets for training, validation in the model, exiting')
    sys.exit(1)    
    
if hasattr(model_module,'data_cast') and hasattr(model_module, 'label_cast'):
    data_cast = model_module.data_cast
    label_cast = model_module.label_cast    
else:
    print("Need to specify how to type output of tensor variables for data and labels")
    sys.exit(1)    

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=1,
                          pin_memory=True)

valid_loader = DataLoader(valid_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=1,
                          pin_memory=True)

if hasattr(model_module, 'num_epochs'):
    num_epochs = model_module.num_epochs
else:
    num_epochs = 20

print("...Checking to see if CUDA  is required")
if hasattr(model_module, 'cuda'):
    import torch.cuda
    cuda = model_module.cuda and torch.cuda.is_available()
    model.cuda()
else:
    cuda = False
   
print("...Training model for ", num_epochs, " epochs (less early stopping)")
start_time = time.time()
prev_time = start_time


for epoch in range(num_epochs):
    
    losses = []
    for batch_idx, (x, y) in enumerate(train_loader):
        
        x, y = data_cast(x), label_cast(y) # needs to be float if regression, long if CrossEntropy
        if cuda:
            x, y = x.cuda(async=True), y.cuda(async=True)
        y_pred = model(x)
        
        loss = training_loss(y_pred, y)
        print(batch_idx, loss.data[0])
        
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optim.zero_grad()
        
        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()
        
        # Clip gradient if specified in model file
        if hasattr(model_module, "clipping"):
            nn.utils.clip_grad_norm(model.parameters(), model_module.clipping)
        
        optim.step()           
        losses.append(loss.data)
        

    epoch_end_time = time.time()
    losses_train.append(losses)
    print("Mean training loss:\t\t {0:.6f}.".format(np.mean(epoch_train_loss))) ### dump these to a text file somewhere else...
    print("Training for epoch ", epoch, " took ", epoch_end_time - epoch_start_time, "s", flush=True)
    
    ### Do we validate?
    if ((epoch + 1) % model_module.validate_every) == 0:
        print("Validating...")
        
        for batch_idx, (x, y) in enumerate(valid_loader):
            
            x, y = data_cast(x), data_cast(y)
            y_pred = model(x)
            
            loss = valid_loss(y_pred, y)
            print("validation batch ", batch_idx, " : ", )
            
         
# tidy up datasets
train_dataset.close()
valid_dataset.close()