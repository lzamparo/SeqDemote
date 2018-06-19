import sys
import os

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from load_pytorch import ATAC_Valid_Dataset
from utils import accounting, train_utils

import importlib.util


### Load up and step through validating data
if len(sys.argv) < 2:
    sys.exit("Usage: pytorch_validate.py <configuration_name> <save_state_file>")

model_config, model_save_file = sys.argv[1], sys.argv[2]
model_path_name = os.path.join(os.path.expanduser(os.getcwd()),'models',model_config)
spec = importlib.util.spec_from_file_location(model_config, model_path_name)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

if hasattr(model_module, 'save_dir') and os.path.exists(os.path.join(train_utils.find_project_root(), 'results')):
    model_save_path = os.path.join(train_utils.find_project_root(), 'results', model_module.save_dir, model_save_file)


print("...Build model and load model state from save state")
model = model_module.net
model.load_state_dict(torch.load(model_save_path, map_location=lambda storage, loc: storage))

if hasattr(model_module, "valid_loss"):
    valid_loss = model_module.valid_loss
    
print("...Loading the data", flush=True)
if hasattr(model_module, 'batch_size'):
    batch_size = model_module.batch_size
else:
    batch_size = 128
    
if hasattr(model_module, 'valid_dataset'):
    valid_dataset = model_module.valid_dataset 
    
valid_loader = DataLoader(valid_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=1,
                          pin_memory=True)

if hasattr(model_module,'data_cast') and hasattr(model_module, 'label_cast'):
    data_cast = model_module.data_cast
    label_cast = model_module.label_cast 

# evaluate the validation data
print("Validating...")
valid_outputs = []
valid_labels = []

losses = []

model.eval()
for batch_idx, (x, y) in enumerate(valid_loader):
    valid_labels.append(y.numpy())
    x, y = data_cast(x), data_cast(y)         
    y_pred = model(x)
    
    loss = valid_loss(y_pred, y)
    if (batch_idx + 1) % 50 == 0:
        print("validation batch ", batch_idx, " : ", loss.data)
    losses.append(loss.data)
    valid_outputs.append(y_pred.data.numpy())
    
print("Mean validation loss:\t\t {0:.6f}".format(np.mean(np.array(losses))))
aupr = train_utils.mt_precision(np.vstack(valid_labels), np.vstack(valid_outputs))
auroc = train_utils.mt_accuracy(np.vstack(valid_labels), np.vstack(valid_outputs))
print("    validation roc:\t {0:.4f}.".format(auroc * 100))
print("    validation aupr:\t {0:.4f}.".format(aupr * 100))