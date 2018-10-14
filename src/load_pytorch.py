from torch.utils.data import Dataset
import h5py
import numpy as np


class BindspaceProbeDataset(Dataset):
    """ Operate on the pseudo-probes from the K562 dataset.   """
    
    def __init__(self, h5_filepath, dataset='training', transform=None):
        path_dict = {'training': ('/data/training/train_data','/labels/training/train_labels'),
                          'validation': ('/data/validation/valid_data','/labels/validation/valid_labels')}
        self.data_path, self.label_path = path_dict[dataset]
        
        self.embedding_dims = 300
        self.h5f = h5py.File(h5_filepath, 'r', libver='latest', swmr=True)
        self.num_peaks, self.rasterized_length = self.h5f[self.data_path].shape
        self.probes_per_peak = self.rasterized_length // self.embedding_dims
        self.num_entries = self.num_peaks * self.probes_per_peak
        self.transform = transform
        TF_overlaps = [s.encode('utf-8') for s in ["CEBPB","CEBPG", "CREB3L1", "CTCF",
                                                   "CUX1","ELK1","ETV1","FOXJ2","KLF13",
                                                   "KLF16","MAFK","MAX","MGA","NR2C2",
                                                   "NR2F1","NR2F6","NRF1","PKNOX1","ZNF143"]]
        TF_colnames = self.h5f[self.label_path].attrs['column_names']
        self.TF_mask_array = np.array([n in TF_overlaps for n in TF_colnames])
        
    def __getitem__(self, index):
        
        peak = index // self.probes_per_peak
        probe = index // self.num_peaks
        start = probe * self.embedding_dims
        stop = (probe + 1) * self.embedding_dims
        
        print("DEBUG: from index ", index, " getting from peak ", peak, " : ", start, " -> ", stop)
        features = self.h5f[self.data_path][peak][start:stop]
        labels = self.h5f[self.label_path][peak]
        if self.transform is not None:
            features = self.transform(features)
        labels = labels[self.TF_mask_array]
        return features, labels
    
    def __len__(self):
        return self.num_entries
    
    def close(self):
        self.h5f.close()
    
    

class Embedded_k562_ATAC_train_dataset(Dataset):
    """ Load up Han's embedded k562 ATAC data for training """
    
    def __init__(self, h5_filepath, transform=None):
        
        self.embedding_dims = 300
        self.h5f = h5py.File(h5_filepath, 'r', libver='latest', swmr=True)
        self.num_entries, self.rasterized_length = self.h5f['/data/training/train_data'].shape
        self.transform = transform
        TF_overlaps = [s.encode('utf-8') for s in ["CEBPB","CEBPG", "CREB3L1", "CTCF",
                                                   "CUX1","ELK1","ETV1","FOXJ2","KLF13",
                                                   "KLF16","MAFK","MAX","MGA","NR2C2",
                                                   "NR2F1","NR2F6","NRF1","PKNOX1","ZNF143"]]
        TF_colnames = self.h5f['/labels/training/train_labels'].attrs['column_names']
        self.TF_mask_array = np.array([n in TF_overlaps for n in TF_colnames])
        
    def __getitem__(self, index):
        
        features = self.h5f['/data/training/train_data'][index]
        labels = self.h5f['/labels/training/train_labels'][index]
        if self.transform is not None:
            features = self.transform(features)
        labels = labels[self.TF_mask_array]
        return features, labels
    
    def __len__(self):
        return self.num_entries
    
    def close(self):
        self.h5f.close()
        
class Embedded_k562_ATAC_validation_dataset(Dataset):
    """ Load up Han's embedded k562 ATAC data for validation """
    
    def __init__(self, h5_filepath, transform=None, TF_overlaps=None):
        
        self.embedding_dims = 300
        self.h5f = h5py.File(h5_filepath, 'r', libver='latest', swmr=True)
        self.num_entries, self.rasterized_length = self.h5f['/data/validation/valid_data'].shape
        self.transform = transform
        if not TF_overlaps:
            TF_overlaps = [s.encode('utf-8') for s in ["CEBPB","CEBPG", "CREB3L1", "CTCF",
                                                   "CUX1","ELK1","ETV1","FOXJ2","KLF13",
                                                   "KLF16","MAFK","MAX","MGA","NR2C2",
                                                   "NR2F1","NR2F6","NRF1","PKNOX1","ZNF143"]]
        else:
            TF_overlaps = [s.encode('utf-8') for s in TF_overlaps]
        TF_colnames = self.h5f['/labels/training/train_labels'].attrs['column_names']
        self.TF_mask_array = np.array([n in TF_overlaps for n in TF_colnames])        
        
    def __getitem__(self, index):
        
        features = self.h5f['/data/validation/valid_data'][index]
        labels = self.h5f['/labels/validation/valid_labels'][index]
        if self.transform is not None:
            features = self.transform(features)
        labels = labels[self.TF_mask_array]
        return features, labels
    
    def __len__(self):
        return self.num_entries
    
    def close(self):
        self.h5f.close()

class ATAC_Train_Dataset(Dataset):
    """ Load the training data set.  Multiple workers need to use
    forked processes, so stateful stuff from the input is not passed on (maybe??)"""

    def __init__(self, h5_filepath, transform=None):
    
        self.h5f = h5py.File(h5_filepath, 'r', libver='latest', swmr=True)
        self.num_entries = self.h5f['/data/train_in'].shape[0]
        self.transform = transform

    def __getitem__(self, index):
        
        features = self.h5f['/data/train_in'][index]
        label = self.h5f['/labels/train_out'][index]
        if self.transform is not None:
            features = self.transform(features)
        return features, label

    def __len__(self):
        return self.num_entries
    
    
    def close(self):
        self.h5f.close()
        
        
class ATAC_Test_Dataset(Dataset):
    """ Load the test data set.  """
    
    def __init__(self, h5_filepath, transform=None):
    
        self.h5f = h5py.File(h5_filepath, 'r', libver='latest', swmr=True)
        self.num_entries = self.h5f["/data/test_in"].shape[0]
        self.transform = transform

    def __getitem__(self, index):
        
        features = self.h5f["/data/test_in"][index]
        label = self.h5f["/labels/test_out"][index]
        if self.transform is not None:
            features = self.transform(features)
        return features, label

    def __len__(self):
        return self.num_entries
    
    
    def close(self):
        self.h5f.close()        
        
        
class ATAC_Valid_Dataset(Dataset):
    """ Load the test data set.  """
    
    def __init__(self, h5_filepath, transform=None):
    
        self.h5f = h5py.File(h5_filepath, 'r', libver='latest', swmr=True)
        self.num_entries = self.h5f["/data/valid_in"].shape[0]
        self.transform = transform

    def __getitem__(self, index):
        
        features = self.h5f["/data/valid_in"][index]
        label = self.h5f["/labels/valid_out"][index]
        if self.transform is not None:
            features = self.transform(features)
        return features, label

    def __len__(self):
        return self.num_entries
    
    
    def close(self):
        self.h5f.close()         


class DNase_Train_Dataset(Dataset):
    
    def __init__(self, h5_filepath, transform=None):
        
        self.h5f = h5py.File(h5_filepath, 'r')
        self.num_entries = self.h5f['/train_in'].shape[0]
        self.transform = transform
        
    def __getitem__(self, index):
        
        features = self.h5f['/train_in'][index]
        label = self.h5f['/train_out'][index]
        if self.transform is not None:
            features = self.transform(features)
        return features, label
    
    def __len__(self):
        return self.num_entries
    
    def close(self):
        self.h5f.close()
        
class DNase_Valid_Dataset(Dataset):
    
    def __init__(self, h5_filepath, transform=None):
        
        self.h5f = h5py.File(h5_filepath, 'r')
        self.num_entries = self.h5f['/valid_in'].shape[0]
        self.transform = transform
        
    def __getitem__(self, index):
        
        features = self.h5f['/valid_in'][index]
        label = self.h5f['/valid_out'][index]
        if self.transform is not None:
            features = self.transform(features)
        return features, label
    
    def __len__(self):
        return self.num_entries
    
    def close(self):
        self.h5f.close()
        
class ProbeReshapeTransformer(object):
    """ Reshape the 300 dimensional probe embedding in a PyTorch friendly
    Tensor shape """
    
    def __init__(self, *args, **kwargs):
        self.probe_dim = 300
    
    def __call__(self, probe):
        return probe.reshape((1,self.probe_dim))
    

class EmbeddingReshapeTransformer(object):
    """ Reshapes the rasterized embedded ATAC-seq windows using the sequence length
    and dimensional embedding. """ 
    
    def __init__(self, embedding_dim, sequence_length):
        assert isinstance(embedding_dim, int)
        assert isinstance(sequence_length, int)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        
    def __call__(self, embedded_rasterized):
        rasterized_length = max(embedded_rasterized.shape)
        return embedded_rasterized.reshape((1,rasterized_length // self.embedding_dim, self.embedding_dim))
        
        
        
class SubsequenceTransformer(object):
    """ extract and sub-sample a sequence of given length after
    accounting for padding.  """
    
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def get_subsequence(self, sequence):
        ''' Helper function for subsampling '''
        
        _, _, cols = sequence.shape
        start = np.random.randint(0, cols - self.output_size) if cols - self.output_size > 0 else 0
        end = start + self.output_size
        subseq = sequence[:,:,start:end]
        return subseq        
        
    
    def __call__(self, sequence):
        ''' Get the subsequences '''
        
        assert(sequence.shape[-1] >= self.output_size)
        
        # trim the padding from peaks if there is any
        if -1.0 in sequence:
            pad_start = np.argmin(sequence)
            return self.get_subsequence(sequence[:,:,0:pad_start])
        else:
            return self.get_subsequence(sequence)
                
            
            
    