import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from nose.tools import eq_, ok_ 

from utils.train_utils import find_project_root
from load_pytorch import ATAC_Valid_Dataset, SubsequenceTransformer
from load_pytorch import Embedded_k562_ATAC_train_dataset
from load_pytorch import Embedded_k562_ATAC_validation_dataset
from load_pytorch import EmbeddingReshapeTransformer
from load_pytorch import ProbeReshapeTransformer
from load_pytorch import BindspaceProbeDataset



'''
The mouse ASA data for CD8+ effector
/                        Group
/data                    Group
/data/test_in            Dataset {12422, 4, 1, 300}
/data/train_in           Dataset {199078, 4, 1, 300}
/data/valid_in           Dataset {6818, 4, 1, 300}
/labels                  Group
/labels/test_out         Dataset {12422, 5}
/labels/train_out        Dataset {199078, 5}
/labels/valid_out        Dataset {6818, 5}
'''

'''
The K562 embedded data, complete with overlap of ChIP-seq peaks
/                        Group
/data                    Group
/data/training           Group
/data/training/train_data Dataset {15584, 84300}
/data/validation         Group
/data/validation/valid_data Dataset {1687, 84300}
/labels                  Group
/labels/training         Group
/labels/training/train_labels Dataset {15584, 209}
/labels/validation       Group
/labels/validation/valid_labels Dataset {1687, 209}
'''

path = os.path.join(find_project_root(), "data", "ATAC", "mouse_asa", "mouse_asa_2k.h5")
valid_examples = 6362
batch_size = 128
subsequence_size = 200
num_batches = valid_examples // batch_size

k562_path = os.path.join(find_project_root(), "data", "ATAC", "K562", "K562_embed_TV_annotated_split.h5")
k562_valid_peak_examples = 1687
k562_valid_probe_examples = 1687 * 281
embedding_dim = 300
sequence_length = 84300
k562_batch_size = 64
k562_targets = 19
k562_num_peak_batches = k562_valid_peak_examples // k562_batch_size
k562_num_probe_batches = k562_valid_probe_examples // k562_batch_size

def setup_dataset_and_loader(transform=False, workers=1):
    if transform:
        transformer = SubsequenceTransformer(subsequence_size)
        valid_dataset = ATAC_Valid_Dataset(path,transform=transformer)
    else:
        valid_dataset = ATAC_Valid_Dataset(path)
        
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers)
    return valid_dataset, valid_loader


def setup_k562_dataset_and_loader(transform=False, workers=1):
    if transform:
        transformer = EmbeddingReshapeTransformer(embedding_dim, 
                                                 sequence_length)
        valid_dataset = Embedded_k562_ATAC_validation_dataset(k562_path, transform=transformer)
    else:
        valid_dataset = Embedded_k562_ATAC_validation_dataset(k562_path)
    
    valid_loader = DataLoader(valid_dataset,
                              batch_size=k562_batch_size,
                              shuffle=True,
                              num_workers=workers)
    return valid_dataset, valid_loader

def setup_k562_bindspace_probe_dataset_and_loader(transform=False, workers=1):
    if transform:
        transformer = ProbeReshapeTransformer()
        valid_dataset = BindspaceProbeDataset(k562_path,dataset='validation',transform=transformer)
    else:
        valid_dataset = BindspaceProbeDataset(k562_path,dataset='validation')
        
    valid_loader = DataLoader(valid_dataset,
                              batch_size=k562_batch_size,
                              shuffle=True,
                              num_workers=workers)
    return valid_dataset, valid_loader

##### k562 data loader tests
def test_build_bindspace_probeset_and_loader():
    """ Can I build a dataset for the BindSpace probe data """
    
    valid_dataset, valid_loader = setup_k562_bindspace_probe_dataset_and_loader(transform=False)
    valid_len = len(valid_dataset)
    valid_dataset.close()
    eq_(valid_len, k562_valid_probe_examples)
    
    
def test_build_bindspace_probeset_and_loader_with_transform():
    """ Can I build a dataset for the BindSpace probe data with the transformer """
    
    valid_dataset, valid_loader = setup_k562_bindspace_probe_dataset_and_loader(transform=True)
    valid_len = len(valid_dataset)
    valid_dataset.close()
    eq_(valid_len, k562_valid_probe_examples)

def test_build_bindspace_probeset_and_loader_shape():
    """ Can I build a dataset for the BindSpace probe data 
    and have it give me the right shaped data """
    
    valid_dataset, valid_loader = setup_k562_bindspace_probe_dataset_and_loader()
    stop = 1
    for batch_idx, (x, y) in enumerate(valid_loader):
        data_shape, target_shape = x.shape, y.shape
        if batch_idx == stop:
            break
    
    valid_dataset.close()
    eq_(data_shape, (k562_batch_size, embedding_dim))
    eq_(target_shape, (k562_batch_size,k562_targets))
    
def test_build_bindspace_probeset_and_loader_shape_with_transform():
    """ Can I build a dataset for the BindSpace probe data 
    and have it give me the right shaped data """
    
    valid_dataset, valid_loader = setup_k562_bindspace_probe_dataset_and_loader(transform=True)
    stop = 1
    for batch_idx, (x, y) in enumerate(valid_loader):
        data_shape, target_shape = x.shape, y.shape
        if batch_idx == stop:
            break
    
    valid_dataset.close()
    eq_(data_shape, (k562_batch_size, 1, embedding_dim))
    eq_(target_shape, (k562_batch_size,k562_targets))

def test_iterate_bindspace_data_loader():
    """ Iterate throught the dataloader backed by 
    the dataset, make sure the number of points seen
    is in the neighbourhood of what's correct.
    """
    valid_dataset, valid_loader = setup_k562_bindspace_probe_dataset_and_loader(transform=True)

    torch.manual_seed(0)
    num_epochs = 1
    data_seen = 0
    lower_bound = k562_batch_size * (k562_num_probe_batches - 1)

    for epoch in range(num_epochs):
        for batch_idx, (x, y) in enumerate(valid_loader):
    
            x, y = Variable(x), Variable(y)
            
            ok_(batch_idx <= num_batches)
            data_seen += x.size()[0]
            
            if batch_idx % 10 == 0:
                print('Epoch:', epoch+1, end='')
                print(' | Batch index:', batch_idx, end='')
                print(' | Batch size:', y.size()[0])
                
        ok_(lower_bound <= data_seen <= k562_valid_probe_examples) 
        
    valid_dataset.close()

def test_build_k562_dataset_and_loader():
    """ Can I build a dataset for the k562 ATAC data
    with the correct shapes """
    
    valid_dataset, valid_loader = setup_k562_dataset_and_loader()
    valid_len = len(valid_dataset)
    valid_dataset.close()
    eq_(valid_len, k562_valid_peak_examples)
    
def test_build_k562_dataset_and_loader_with_transform():
    """ Can I build a dataset for the k562 ATAC data
    with the correct shapes """
    
    valid_dataset, valid_loader = setup_k562_dataset_and_loader(transform=True)
    valid_len = len(valid_dataset)
    valid_dataset.close()
    eq_(valid_len, k562_valid_peak_examples)    
    
def test_iterate_k562_data_loader():
    """ Iterate throught the dataloader backed by 
    the dataset, make sure the number of points seen
    is in the neighbourhood of what's correct.
    """
    valid_dataset, valid_loader = setup_k562_dataset_and_loader(transform=True)

    torch.manual_seed(0)
    num_epochs = 1
    data_seen = 0
    lower_bound = k562_batch_size * (k562_num_peak_batches - 1)

    for epoch in range(num_epochs):
        for batch_idx, (x, y) in enumerate(valid_loader):
    
            x, y = Variable(x), Variable(y)
            
            ok_(batch_idx <= num_batches)
            data_seen += x.size()[0]
            
            if batch_idx % 10 == 0:
                print('Epoch:', epoch+1, end='')
                print(' | Batch index:', batch_idx, end='')
                print(' | Batch size:', y.size()[0])
                
        ok_(lower_bound <= data_seen <= k562_valid_peak_examples) 
        
    valid_dataset.close()
    
##### ATAC data loader tests

def test_build_data_loader():
    """ Can I build a dataset for the ATAC data with 
    the correct length """
    
    valid_dataset, valid_loader = setup_dataset_and_loader()
    valid_len = len(valid_dataset)
    valid_dataset.close()
    eq_(valid_len, valid_examples)


def test_transformer_only():
    """ Transformer seems to be acting up occasionally, why? """
    valid_dataset, valid_loader = setup_dataset_and_loader()
    transformer = SubsequenceTransformer(subsequence_size)
    
    torch.manual_seed(0)
    data_seen = 0
    lower_bound = batch_size * (num_batches - 1)
    for batch_idx, (x, y) in enumerate(valid_loader):
        
        for seq in x:
            try:
                x_trans = transformer(seq)
                ok_(x_trans.size()[-1] == subsequence_size)
            except:
                print("Encountered a transformer error for: ", seq)
                
        data_seen += x.size()[0]
    print("lower bound, ", lower_bound, " data_seen, ", data_seen, " valid_examples ", valid_examples)            
    ok_(lower_bound <= data_seen <= valid_examples)
    valid_dataset.close()
            

def test_lengths():
    """ Something funny going on with sequence lengths """
    
    valid_dataset, valid_loader = setup_dataset_and_loader()
    
    for batch_idx, (x, y) in enumerate(valid_loader):
        for seq in x:
            ok_(seq.shape[-1] > 0)
            ok_(seq.shape[-1] >= 200)
            
    valid_dataset.close()
            

def test_transform_data_loader():
    """ Make sure the dataset has the correct 
    transformed lengths """

    valid_dataset, valid_loader = setup_dataset_and_loader(transform=True)

    torch.manual_seed(0)
    data_seen = 0
    lower_bound = batch_size * (num_batches - 1)

    for batch_idx, (x, y) in enumerate(valid_loader):

        x, y = Variable(x), Variable(y)
        ok_(x.size()[-1] == subsequence_size)
        data_seen += x.size()[0]
            
    ok_(lower_bound <= data_seen <= valid_examples)
    valid_dataset.close()
    
    
def test_iterate_data_loader():
    """ Iterate throught the dataloader backed by 
    the dataset, make sure the number of points seen
    is in the neighbourhood of what's correct.
    """
    valid_dataset, valid_loader = setup_dataset_and_loader()

    torch.manual_seed(0)
    num_epochs = 1
    data_seen = 0
    lower_bound = 128 * (num_batches - 1)

    for epoch in range(num_epochs):
        for batch_idx, (x, y) in enumerate(valid_loader):
    
            x, y = Variable(x), Variable(y)
            
            ok_(batch_idx <= num_batches)
            data_seen += x.size()[0]
            
            if batch_idx % 10 == 0:
                print('Epoch:', epoch+1, end='')
                print(' | Batch index:', batch_idx, end='')
                print(' | Batch size:', y.size()[0])
                
        ok_(lower_bound <= data_seen <= valid_examples) 
        
    valid_dataset.close()

    

