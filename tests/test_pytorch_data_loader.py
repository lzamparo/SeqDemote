import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from nose.tools import eq_, ok_ 

from utils.train_utils import find_project_root
from load_pytorch import ATAC_Valid_Dataset, SubsequenceTransformer

'''
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

path = os.path.join(find_project_root(), "data", "ATAC", "mouse_asa", "mouse_asa.h5")
valid_examples = 6818
batch_size = 128
subsequence_size = 200
num_batches = valid_examples // batch_size

def setup_dataset_and_loader(transform=False):
    if transform:
        transformer = SubsequenceTransformer(subsequence_size)
        valid_dataset = ATAC_Valid_Dataset(path,transform=transformer)
    else:
        valid_dataset = ATAC_Valid_Dataset(path)
        
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)
    return valid_dataset, valid_loader


def test_build_data_loader():
    """ Can I build a dataset the ATAC data with 
    the correct length """
    
    valid_dataset, valid_loader = setup_dataset_and_loader()
    valid_len = len(valid_dataset)
    valid_dataset.close()
    eq_(valid_len, valid_examples)


def test_transform_data_loader():
    """ Make sure the dataset has the correct 
    transformed lengths """

    valid_dataset, valid_loader = setup_dataset_and_loader(transform=True)

    torch.manual_seed(0)
    data_seen = 0
    lower_bound = 128 * (num_batches - 1)

    for batch_idx, (x, y) in enumerate(valid_loader):

        x, y = Variable(x), Variable(y)
        ok_(x.size()[-1] == subsequence_size)
        data_seen += x.size()[0]
            
    ok_(lower_bound <= data_seen <= valid_examples)     
    
    
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

    

