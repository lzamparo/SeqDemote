import sys
import argparse
import os
import torch
import importlib.util

import numpy as np
from torch.utils.data import DataLoader
from utils import train_utils


parser = argparse.ArgumentParser()
parser.add_argument("config", help='path to config file')
parser.add_argument("--savefile", help='path to saved model state', default=None)
args = parser.parse_args()

### Load up and step through validating data
model_config = args.config
model_save_path_present = False
if args.savefile:
    model_save_file = args.savefile
    
model_path_name = os.path.join(os.path.expanduser(os.getcwd()),'models',model_config)
spec = importlib.util.spec_from_file_location(model_config, model_path_name)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

if args.savefile and hasattr(model_module, 'save_dir') and os.path.exists(os.path.join(train_utils.find_project_root(), 'results')):
    model_save_path = os.path.join(train_utils.find_project_root(), 'results', model_module.save_dir, model_save_file)
    model_loaded_from_save = True

print("...Build model, load model state from save state if provided")
model = model_module.net
if model_save_path_present:
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
valid_losses = []

model.eval()
for batch_idx, (x, y) in enumerate(valid_loader):
    valid_labels.append(y.numpy())
    x, y = data_cast(x), data_cast(y)         
    
    y_pred = model(x)
    losses = train_utils.per_task_loss(y_pred, y, valid_loss, do_sum=False)
    loss = sum(losses)
    if (batch_idx + 1) % 50 == 0:
        print("validation batch ", batch_idx, " : ", loss.data)
    valid_losses.append(losses)
    
    y_pred_repacked = train_utils.repackage_to_cpu(y_pred, unsqueeze=True)
    y_pred_sigmoid = torch.nn.functional.sigmoid(y_pred_repacked)  
    valid_outputs.append(y_pred_sigmoid.data.numpy())
    
print("Mean validation loss:\t\t {0:.6f}".format(np.mean(np.array(losses))))
pr50 = train_utils.mt_precision_at_recall(np.vstack(valid_labels), np.vstack(valid_outputs), average=False)
avg_f1 = train_utils.mt_avg_f1_score(np.vstack(valid_labels), np.vstack(valid_outputs), average=False)
avg_mcc = train_utils.mt_avg_mcc(np.vstack(valid_labels), np.vstack(valid_outputs), average=False)

# Save the results to file
filename = os.path.basename(model_config)
filename = filename.lstrip('BindSpace_').rstrip(".py") + ".txt"
with open(os.path.join(train_utils.find_project_root(), 'results' , 'BindSpace_embedding_extension',filename),'w') as f:
    for e in pr50:
        print(e, file=f)
