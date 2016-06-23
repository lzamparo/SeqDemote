import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

import lasagne as nn

import importlib.util
import pickle
from datetime import datetime, timedelta
import string

import matplotlib
matplotlib.use('agg')
import pylab as plt

import generators
from utils import accounting
from utils import train_utils
from utils import network_repr

from subprocess import Popen


if len(sys.argv) < 2:
    sys.exit("Usage: train_convnet.py <configuration_name>")

model_config = sys.argv[1]
model_path_name = os.path.join(os.path.expanduser(os.getcwd()),'models',model_config)
spec = importlib.util.spec_from_file_location(model_config, model_path_name)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
    
expid = accounting.generate_expid(model_config)
metadata_tmp_path = os.path.join(model_path_name, "checkpoints", expid + ".pkl")
print("Experiment ID: ", expid)

print("...Build model")
model = model_module.build_model()
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
print("...layer output shapes:")
print(network_repr.get_network_str(all_layers, get_network=False, incomings=True, 
                                  outgoings=True))


print("...setting up shared vars, building the training & validation objectives ", flush=True)

x_shared = nn.utils.shared_empty(dim=len(l_in.output_shape)) 
y_shared = nn.utils.shared_empty(dim=2)  # 
t = nn.utils.shared_empty(dim=2)  # target shared var per batch

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


print("...setting the learning rate schedule ")
if hasattr(model_module, 'learning_rate_schedule'):
    learning_rate_schedule = model_module.learning_rate_schedule
else:
    learning_rate_schedule = { 0: model_module.learning_rate }
    
learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))


print("...setting the optimization scheme ")
if hasattr(model_module, 'build_updates'):
    updates = model_module.build_updates(train_loss, all_params, learning_rate, model_module.momentum)
else:
    updates = nn.updates.rmsprop(train_loss, all_params, learning_rate, 0.9)

if hasattr(model_module, 'censor_updates'):
    updates = model_module.censor_updates(updates, l_out)

#iter_train = theano.function([idx], train_loss, givens=givens, updates=updates, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
iter_train = theano.function([idx], train_loss, givens=givens, updates=updates)
compute_output = theano.function([idx], valid_output, givens=givens, on_unused_input="ignore")


if hasattr(model_module, 'resume_path'):
    print("Load model parameters for resuming")
    if hasattr(model_module, 'pre_init_path'):
        print("lresume = lout")
        l_resume = l_out
    resume_metadata = np.load(model_module.resume_path)
    nn.layers.set_all_param_values(l_resume, resume_metadata['param_values'])

    start_chunk_idx = resume_metadata['chunks_since_start'] + 1
    chunks_train_idcs = range(start_chunk_idx, model_module.num_chunks_train)

    # set lr to the correct value
    current_lr = np.float32(train_utils.current_learning_rate(learning_rate_schedule, start_chunk_idx))
    print("...setting learning rate to {0:.7f}.".format(current_lr))
    learning_rate.set_value(current_lr)
    losses_train = resume_metadata['losses_train']
    losses_valid_log = resume_metadata['losses_eval_valid']
    losses_valid_auc = resume_metadata['losses_eval_train']
    
elif hasattr(model_module, 'pre_init_path'):
    print("Load model parameters for initializing first x layers")
    resume_metadata = np.load(model_module.pre_init_path)
    nn.layers.set_all_param_values(l_resume, resume_metadata['param_values'][-len(all_excluded_params):])

    chunks_train_idcs = range(model_module.num_chunks_train)
    losses_train = []
    losses_valid_log = []
    losses_valid_auc = []
    losses_valid_aupr = []
    
else:
    chunks_train_idcs = range(model_module.num_chunks_train)
    losses_train = []
    losses_valid_log = []
    losses_valid_auc = []
    losses_valid_aupr = []

print("...Load data", flush=True)
model_module.data_loader.load_train()

if hasattr(model_module, 'resume_path'):
    model_module.data_loader.set_params(resume_metadata['data_loader_params'])

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
start_time = time.time()
prev_time = start_time

copy_process = None
num_batches_chunk = model_module.chunk_size // model_module.batch_size

for epoch in range(num_epochs):

    ### train in chunks
    for e, (x_chunk, y_chunk) in zip(chunks_train_idcs, create_train_gen()):
        print("Chunk ", str(e + 1), " of ", model_module.num_chunks_train, flush=True)
        chunk_start_time = time.time()
        
        if e in learning_rate_schedule:
            lr = np.float32(learning_rate_schedule[e])
            print("...setting learning rate to {0:.7f}.".format(lr))
            learning_rate.set_value(lr)
    
        print("...load training data onto device")
        x_shared.set_value(x_chunk)
        y_shared.set_value(y_chunk)
    
        print("...performing batch SGD")
        losses = []
        for b in range(num_batches_chunk):
            loss = iter_train(b)
            if np.isnan(loss):
                raise RuntimeError("NaN DETECTED.")
            losses.append(loss)
    
            
        mean_train_loss = np.mean(losses)
        print("Mean training loss:\t\t {0:.6f}.".format(mean_train_loss))
        losses_train.append(mean_train_loss)
        chunk_end_time = time.time()
        print("Training for chunk ", e, " took ", chunk_end_time - chunk_start_time, "s", flush=True)
    
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

        outputs = np.vstack(outputs)
        loss = train_utils.log_loss(outputs, np.vstack(labels))
        acc = train_utils.mt_accuracy(outputs, np.vstack(labels))
        precision = train_utils.mt_precision(outputs, np.vstack(labels))
        print("    validation loss:\t {0:.6f}.".format(loss))
        print("    validation roc:\t {0:.2f}.".format(acc * 100))
        print("    validation aupr:\t {0:.2f}".format(precision * 100))
        losses_valid_log.append(loss)
        losses_valid_auc.append(acc)
        losses_valid_aupr.append(precision)
        del outputs
    
    
        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        est_time_left = time_since_start * (float(model_module.num_chunks_train - (e + 1)) / float(e + 1 - chunks_train_idcs[0]))
        eta = datetime.now() + timedelta(seconds=est_time_left)
        eta_str = eta.strftime("%c")
        print(accounting.hms(time_since_start), " since start ", time_since_prev)
        print(" estimated time remaining: ", eta_str)
    
        ### Do we save the model state?
        if ((epoch + 1) % model_module.save_every) == 0:
            print("Saving metadata, parameters")
    
            with open(metadata_tmp_path, 'w') as f:
                pickle.dump({
                    'configuration': config_name,
                    'experiment_id': expid,
                    'chunks_since_start': e,
                    'losses_train': losses_train,
                    'losses_eval_valid': losses_valid_log,
                    'losses_eval_train': losses_valid_auc,
                    'time_since_start': time_since_start,
                    'param_values': nn.layers.get_all_param_values(l_out), 
                    'data_loader_params': model_module.data_loader.get_params(),
                }, f, pickle.HIGHEST_PROTOCOL)
    
            # terminate the previous copy operation if it hasn't finished
            #if copy_process is not None:
            #    copy_process.terminate()
    
            #copy_process = Popen(['cp', metadata_tmp_path, metadata_target_path])
    
            #print("  saved to ", metadata_tmp_path, " copying to ", metadata_target_path)
            #print
