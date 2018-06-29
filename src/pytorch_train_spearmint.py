import sys
import os
import time
import pickle

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from load_pytorch import ATAC_Train_Dataset, ATAC_Valid_Dataset
from load_pytorch import SubsequenceTransformer
from utils import accounting, train_utils

import importlib.util
from datetime import datetime, timedelta
import string
import simple_spearmint

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

    
print("Experiment ID: ", expid)
model_hyperparams_dict = model_module.model_hyperparams_dict
ss = simple_spearmint.SimpleSpearmint(model_hyperparams_dict, minimize=False)


def validation_aupr_objective(suggestion, model_module):
    ''' Instantiate the network in the model_module, with 
    hyperparameters for optimizer give by suggestion '''
    
    print("...Build model")
    model = model_module.reinitialize_model()
    
    print("...number of parameters: ", train_utils.count_params(model.parameters()))
    print("...layer output shapes:")
    print(model)
    
    print("...setting up loss functions ", flush=True)
    training_loss = model_module.train_loss
    valid_loss = model_module.valid_loss
        
    print("...setting the learning rate schedule ", flush=True)
    if hasattr(model_module, 'learning_rate_schedule'):
        learning_rate_schedule = model_module.learning_rate_schedule
    else:
        learning_rate_schedule = { 0: model_module.learning_rate }
        
    print("...Checking to see if CUDA  is required", flush=True)
    if hasattr(model_module, 'cuda'):
        cuda = model_module.cuda and torch.cuda.is_available()
    else:
        cuda = False
        
    if cuda:
        import torch.cuda
        model.cuda()    
        
    print("...setting up optimizer ", flush=True)
    
    weights, biases, sparse_weights = model_module.get_model_param_lists(model)
    optimizer, opd_list = model_module.initialize_optimizer(weights, 
                                                            biases, 
                                                            sparse_weights,
                                                            suggestion)
    optimizer_kwargs = model_module.optimizer_kwargs
    optim = optimizer(opd_list, **optimizer_kwargs)     
    
    additional_losses = model_module.get_additional_losses(model, suggestion)
    optim.zero_grad()
    
    print("...setting up logging for losses ")
    losses_train = []
    losses_valid_log = []
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
       
    print("...Training model for ", num_epochs, " epochs")
    start_time = time.time()
    prev_time = start_time
    
    for epoch in range(num_epochs):
        
        print("Starting training for epoch ", epoch)
        model.train()  # set model to training, if not already.
        epoch_losses = []
        epoch_start_time = time.time()
        total_batches = len(train_loader)
        for batch_idx, (x, y) in enumerate(train_loader):
            
            x, y = data_cast(x), label_cast(y) # needs to be float if regression, long if CrossEntropy
            if cuda:
                x, y = x.cuda(async=True), y.cuda(async=True)
            y_pred = model(x)
            
            losses = train_utils.per_task_loss(y_pred, y, training_loss,do_sum=False)
            for reg in additional_losses:
                losses.append(reg)
            
            loss = sum(losses)   
            if (batch_idx + 1) % 100 == 0:
                print('Epoch [{}/{}], batch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 
                                                                          num_epochs, 
                                                                          batch_idx+1, 
                                                                          total_batches, 
                                                                          loss.item()))
            
            optim.zero_grad()
            loss.backward(retain_graph=True)
            
            # Clip gradient if specified in model file
            if hasattr(model_module, "clipping"):
                nn.utils.clip_grad_norm_(model.parameters(), model_module.clipping)
            
            optim.step()           
            epoch_losses.append(loss.data)
            
    
        epoch_end_time = time.time()
        losses_train.append(epoch_losses)
        print("Mean training loss:\t\t {0:.6f}.".format(np.mean(np.array(epoch_losses)))) 
        print("Training for epoch ", epoch, " took ", epoch_end_time - epoch_start_time, "s", flush=True)
        
        ### Do we validate?
        if ((epoch + 1) % model_module.validate_every) == 0:
            print("Validating...")
            valid_outputs = []
            valid_labels = []
            losses = []
            
            model.eval()  # set model to evaluation mode: turn off dropout
            
            for batch_idx, (x, y) in enumerate(valid_loader):
                valid_labels.append(y.numpy())
                x, y = data_cast(x), data_cast(y)
                if cuda:
                    x, y = x.cuda(async=True), y.cuda(async=True)            
                y_pred = model(x)
                
                loss = valid_loss(y_pred, y)
                if (batch_idx + 1) % 10 == 0:
                    print("validation batch ", batch_idx, " : ", loss.data)
                losses.append(loss.data)
                if cuda:
                    y_pred = y_pred.cpu()
                valid_outputs.append(y_pred.data.numpy())
                
            aupr = train_utils.mt_precision(np.vstack(valid_labels), np.vstack(valid_outputs))
            print("    validation aupr:\t {0:.4f}.".format(aupr * 100))
            losses_valid_aupr.append(aupr)
            
            
            
            
    return max(losses_valid_aupr)     


# Seed with 5 randomly chosen parameter settings
for n in range(5):
    # Get random parameter settings
    suggestion = ss.suggest_random()
    
    # Retrieve an objective value for these parameters
    value = validation_aupr_objective(suggestion, model_module)
    print("Random trial {}: {} -> {}".format(n + 1, suggestion, value))
    
    # Update the optimizer on the result
    ss.update(suggestion, value)

# Run for 20 hyperparameter optimization trials
for n in range(20):
    
    # Get a suggestion from the optimizer
    suggestion = ss.suggest()
    
    # Get an objective value
    value = validation_aupr_objective(suggestion, model_module)
    print("GP trial {}: {} -> {}".format(n + 1, suggestion, value))
    
    # Update the optimizer on the result
    ss.update(suggestion, value)
    best_parameters, best_objective = ss.get_best_parameters()
    print("Best parameters {} for objective {}".format(best_parameters, best_objective))
        
# tidy up datasets
train_dataset.close()
valid_dataset.close()
