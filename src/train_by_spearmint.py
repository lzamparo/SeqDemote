import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
#from theano.compile.nanguardmode import NanGuardMode
import numpy as np

import lasagne as nn

import importlib.util

from datetime import datetime, timedelta
import pickle
import string

import matplotlib
matplotlib.use('agg')
import pylab as plt

import generators
from utils import accounting
from utils import train_utils
from utils import network_repr

import simple_spearmint

if len(sys.argv) < 2:
    sys.exit("Usage: train_by_spearmint.py <configuration_name>")

model_config = sys.argv[1]
model_path_name = os.path.join(os.path.expanduser(os.getcwd()),'models',model_config)
spec = importlib.util.spec_from_file_location(model_config, model_path_name)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
    
expid = accounting.generate_expid(model_config)
expid = expid.split('/')[-1]
print("Experiment ID: ", expid)

print("...get the relevant model paremeter search space, set up spearmint")
model_params_dict = model_module.model_params_dict
ss = simple_spearmint.SimpleSpearmint(model_params_dict, minimize=False)

# Define an objective function, must return a scalar value
def objective(params_dict, model_module):
    
    try:
        print("...Build model")
        model = model_module.build_model(params_dict)
    except:
        print("Model configuration was invalid, trying another one...")
        return 0.1

    if len(model) == 4:
        l_in, l_out, l_resume, l_exclude = model
    elif len(model) == 3:
        l_in, l_out, l_resume = model
        l_exclude = l_in
    else:
        l_in, l_out = model
        l_resume = l_out
        l_exclude = l_in


    all_layers = nn.layers.get_all_layers(l_out)
    num_params = nn.layers.count_params(l_out)

    print("...number of parameters: ", num_params)

    x_shared = nn.utils.shared_empty(dim=len(l_in.output_shape)) 
    y_shared = nn.utils.shared_empty(dim=2)   
    t = nn.utils.shared_empty(dim=2)                                ### target shared var per batch
    valid_output = nn.layers.get_output(l_out, deterministic=True)  ### no dropout for validation 

    idx = T.lscalar('idx')

    givens = {
        t: y_shared[idx*model_module.batch_size:(idx+1)*model_module.batch_size],
        l_in.input_var: x_shared[idx*model_module.batch_size:(idx+1)*model_module.batch_size],
    }

    if hasattr(model_module, 'build_objective'):
        train_loss = model_module.build_objective(l_in, l_out, t, training_mode=True)
    else:
        train_loss = nn.objectives.aggregate(nn.objectives.binary_crossentropy(l_out, t))

    all_excluded_params = nn.layers.get_all_params(l_exclude)
    all_params = nn.layers.get_all_params(l_out)
    all_params = list(set(all_params) - set(all_excluded_params))

    if hasattr(model_module, 'learning_rate_schedule'):
        learning_rate_schedule = model_module.learning_rate_schedule
    else:
        learning_rate_schedule = { 0: model_module.learning_rate }
        
    learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))

    if hasattr(model_module, 'build_updates'):
        updates = model_module.build_updates(train_loss, all_params, learning_rate, model_module.momentum)
    else:
        updates = nn.updates.rmsprop(train_loss, all_params, learning_rate, 0.9)

    if hasattr(model_module, 'censor_updates'):
        updates = model_module.censor_updates(updates, l_out)

    iter_train = theano.function([idx], train_loss, givens=givens, updates=updates)
    compute_output = theano.function([idx], valid_output, givens=givens, on_unused_input="ignore")   
    chunks_train_idcs = range(model_module.num_chunks_train)
    losses_valid_auc = []
    losses_valid_aupr = []
    losses_train = []
    losses_valid_log = []

    model_module.data_loader.load_train()

    if hasattr(model_module, 'task_type'):
        task_type = model_module.task_type
    else:
        task_type = 'mt_classification'

    if hasattr(model_module, 'create_train_gen'):
        create_train_gen = model_module.create_train_gen
    else:
        create_train_gen = lambda: model_module.data_loader.create_batch_gen()

    if hasattr(model_module, 'create_valid_gen'):
        create_valid_gen = model_module.create_valid_gen
    else:
        create_valid_gen = lambda: model_module.data_loader.create_valid_gen()

    if hasattr(model_module, 'num_epochs'):
        num_epochs = model_module.num_epochs
    else:
        num_epochs = 15
       
    print("...Training model for ", num_epochs, " epochs (less early stopping)")
    num_batches_chunk = model_module.chunk_size // model_module.batch_size

    for epoch in range(num_epochs):

        ### train in chunks
        epoch_train_loss = []
        epoch_start_time = time.time()
        for e, (x_chunk, y_chunk) in zip(chunks_train_idcs, create_train_gen()):
            
            if e in learning_rate_schedule:
                lr = np.float32(learning_rate_schedule[e])
                learning_rate.set_value(lr)
                
            x_shared.set_value(x_chunk)
            y_shared.set_value(y_chunk)
            losses = []
            for b in range(num_batches_chunk):
                loss = iter_train(b)
                outs = compute_output(b)
                if np.isnan(loss):
                    raise RuntimeError("NaN DETECTED.")
                losses.append(loss)
            
            mean_train_loss = np.mean(losses)
            epoch_train_loss.append(mean_train_loss)
            
        epoch_end_time = time.time()
        losses_train.append(epoch_train_loss)
        print("Mean training loss:\t\t {0:.6f}.".format(np.mean(epoch_train_loss))) ### dump these to a text file somewhere else...
        print("Training for epoch ", epoch, " took ", epoch_end_time - epoch_start_time, "s", flush=True)
        
        ### Do we validate?
        if ((epoch + 1) % model_module.validate_every) == 0:
            print("Validating...")
            
            outputs = []
            labels = []
            for x_chunk_valid, y_chunk_valid in create_valid_gen():
                num_batches_chunk_valid = x_chunk_valid.shape[0] // model_module.batch_size

                x_shared.set_value(x_chunk_valid)

                outputs_chunk = []
                for b in range(num_batches_chunk_valid):
                    out = compute_output(b)
                    outputs_chunk.append(out)

                outputs_chunk = np.vstack(outputs_chunk)
                #outputs_chunk = outputs_chunk[:chunk_length_eval] # truncate to the right length
                outputs.append(outputs_chunk)
                labels.append(y_chunk_valid)


            outputs = np.vstack(outputs)                            ### dump these to a list, pickle the list
            loss = train_utils.log_loss(outputs, np.vstack(labels))
            if task_type != 'mt_classificaiton':
                acc = train_utils.st_accuracy(outputs, np.vstack(labels))
            else:
                acc = train_utils.mt_accuracy(outputs, np.vstack(labels))
                precision = train_utils.mt_avg_precision(outputs, np.vstack(labels))
            print("    validation loss:\t {0:.6f}.".format(loss))  ### dump these to a text file somewhere else
            print("    validation roc:\t {0:.2f}.".format(acc * 100))
            losses_valid_log.append(loss)
            losses_valid_auc.append(acc)
            del outputs
            
    return max(losses_valid_auc)


# Seed with 5 randomly chosen parameter settings
# (this step is optional, but can be beneficial)
for n in range(5):
    # Get random parameter settings
    suggestion = ss.suggest_random()
    
    # Retrieve an objective value for these parameters
    value = objective(suggestion, model_module)
    print("Random trial {}: {} -> {}".format(n + 1, suggestion, value))
    
    # Update the optimizer on the result
    ss.update(suggestion, value)

# Run for 50 hyperparameter optimization trials
for n in range(50):
    
    # Get a suggestion from the optimizer
    suggestion = ss.suggest()
    
    # Get an objective value
    value = objective(suggestion, model_module)
    print("GP trial {}: {} -> {}".format(n + 1, suggestion, value))
    
    # Update the optimizer on the result
    ss.update(suggestion, value)
    best_parameters, best_objective = ss.get_best_parameters()
    print("Best parameters {} for objective {}".format(best_parameters, best_objective))

         
    
