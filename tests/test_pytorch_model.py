import os
import torch
import importlib
import numpy as np
import pickle
from torch.utils.data import DataLoader
from torch.autograd import Variable


from nose.tools import eq_, ok_ 

from utils.train_utils import find_project_root, mt_avg_precision, mt_accuracy
from load_pytorch import DNase_Valid_Dataset

path = os.path.join(find_project_root(), "data", "DNase", "encode_roadmap.h5")
model_config = os.path.join("torch_models","torch_basset_repro.py")

test_model_pickle_path = os.path.join(find_project_root(),"tests","test_data","pytorch_basset.mdl")
test_losses_pickle_path = os.path.join(find_project_root(),"tests","test_data","pytorch_params.pkl")

valid_examples = 70000
batch_size = 128
subsequence_size = 600
num_batches = valid_examples // batch_size

def reconstitute_model():
    
    model_path_name = os.path.join(find_project_root(), 'src', 'models', model_config)
    spec = importlib.util.spec_from_file_location(model_config, model_path_name)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model = model_module.net    
    return model, model_module

def setup_dataset_and_loader(transform=False, workers=1):
    
    model, model_module = reconstitute_model()
    
    if hasattr(model_module,'data_cast') and hasattr(model_module, 'label_cast'):
        data_cast = model_module.data_cast
        label_cast = model_module.label_cast
    
    valid_loss = model_module.valid_loss
        
    if transform:
        transformer = SubsequenceTransformer(subsequence_size)
        valid_dataset = DNase_Valid_Dataset(path,transform=transformer)
    else:
        valid_dataset = DNase_Valid_Dataset(path)
        
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers)
    return valid_dataset, valid_loader, model, valid_loss, data_cast, label_cast


def test_build_data_loader():
    """ Can I build a dataset the ATAC data with 
    the correct length """
    
    valid_dataset, valid_loader, model, valid_loss, data_cast, label_cast = setup_dataset_and_loader()
    valid_len = len(valid_dataset)
    valid_dataset.close()
    eq_(valid_len, valid_examples)


def test_validation_loop_shapes_and_types():
    """ Ensure predicted tensors are of the correct type,
    and that the predictions are the same shape """
    
    valid_dataset, valid_loader, model, valid_loss, data_cast, label_cast = setup_dataset_and_loader()
    
    torch.manual_seed(0)
    data_seen = 0
    
    valid_outputs = []
    valid_labels = []
    losses = []
    
    for batch_idx, (x, y) in enumerate(valid_loader):
        if batch_idx > 20:
            break
        valid_labels.append(y.numpy())
        x, y = data_cast(x), data_cast(y)          
        y_pred = model(x)
        
        loss = valid_loss(y_pred, y)
        losses.append(loss.data)
        ok_(isinstance(y_pred, torch.autograd.Variable))
        ok_(isinstance(y_pred.data, torch.FloatTensor))
        valid_outputs.append(y_pred.data.numpy())
        
    eq_(np.vstack(valid_labels).shape, np.vstack(valid_outputs).shape)    
    valid_dataset.close()

    
def test_pickle_model():
    ''' running into a load of errors just getting a model to go through the
    step of '''
    
    valid_dataset, valid_loader, model, valid_loss, data_cast, label_cast = setup_dataset_and_loader()
    model_config = "pytorch_models/torch_basset_repro.py"
    expid = "001_test.txt"
    
    torch.manual_seed(0)
    data_seen = 0
    
    valid_outputs = []
    valid_labels = []
    losses_train=[]
    losses_valid_log = []
    losses_valid_aupr = []
    losses_valid_auroc = []
    
    for batch_idx, (x, y) in enumerate(valid_loader):
        if batch_idx > 30:
            break
        valid_labels.append(y.numpy())
        x, y = data_cast(x), data_cast(y)          
        y_pred = model(x)
        
        loss = valid_loss(y_pred, y)
        
        losses_valid_log.append(loss.data)
        ok_(isinstance(y_pred, torch.autograd.Variable))
        ok_(isinstance(y_pred.data, torch.FloatTensor))
        valid_outputs.append(y_pred.data.numpy())    
    
        aupr = mt_avg_precision(np.vstack(valid_labels), np.vstack(valid_outputs))
        auroc = mt_accuracy(np.vstack(valid_labels), np.vstack(valid_outputs))
        losses_valid_aupr.append(aupr)
        losses_valid_auroc.append(auroc)
    
    # serialize the model
    torch.save(model.state_dict(),test_model_pickle_path)
        
    # dump to pickle
    save_dict = {
        'configuration': model_config,
        'experiment_id': expid,
        'losses_train': losses_train,
        'losses_valid_xent': losses_valid_log,
        'losses_valid_auc': losses_valid_auroc,
        'losses_valid_aupr': losses_valid_aupr
    }    
    with open(test_losses_pickle_path, 'wb') as f:
        pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)            
    
    # unpickle, compare model params
    new_model, new_model_module = reconstitute_model()
    new_model.load_state_dict(torch.load(test_model_pickle_path))   
    for old_p, new_p in zip(model.parameters(),new_model.parameters()):
        ok_(np.allclose(old_p.data.numpy(), new_p.data.numpy()))
    
    # unpickle, compare losses
    with open(test_losses_pickle_path, 'rb') as f:
        unpickled_losses = pickle.load(f)
        
    for k in unpickled_losses.keys():
        eq_(save_dict[k], unpickled_losses[k])
        
    valid_dataset.close()