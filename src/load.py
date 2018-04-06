import sys
import numpy as np
import itertools

import generators
import os
import buffering
import h5py
from abc import ABCMeta, abstractmethod

class DataLoader(object):
    __metaclass__ = ABCMeta
    
    params = {} # attributes that need to be stored after training and loaded at test time.

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @abstractmethod
    def load_train(self):   
        pass
    
    @abstractmethod
    def load_test(self):
        pass
        
    @abstractmethod    
    def load_validation(self):
        pass

    def get_params(self):
        return { pname: getattr(self, pname, None) for pname in self.__dict__ }
        
    def set_params(self, p):
        self.__dict__.update(p)
       
    def get_h5_handle(self, path):
        """ Get the h5 handle to the data along this path """
        try:
            my_path = h5py.File(path)
        except:
            print("Error: no valid h5 file at path : ", path)
            sys.exit(1)
        return my_path


class HematopoeticDataLoader(DataLoader):
    
    def __init__(self, **kwargs):
        DataLoader.__init__(self)
        self.__dict__.update(kwargs)
        if not hasattr(self, 'data_path'):
            self.data_path = os.path.abspath("../data/DNase/hematopoetic_data.h5")
        if not hasattr(self, 'peaks_vs_flanks'):
            self.peaks_vs_flanks = True
            
    def load_train(self):
        h5file = self.get_h5_handle(self.data_path)
        peaks_train_in = h5file['/peaks/data/train_in']
        flanks_train_in = h5file['/flanks/data/train_in']
        n_peaks = peaks_train_in.shape[0]
        n_flanks = flanks_train_in.shape[0]
        self.train_set_size = n_peaks + n_flanks
        train_set_shape = tuple([n_peaks + n_flanks, peaks_train_in.shape[1], peaks_train_in.shape[2], peaks_train_in.shape[3]])
        self.train_in = np.zeros(train_set_shape, dtype=peaks_train_in.dtype)
        self.train_in[0:n_peaks,:,:,:] = peaks_train_in[:]
        self.train_in[n_peaks:n_peaks+n_flanks,:,:,:] = flanks_train_in[:]
        
        peaks_train_out = h5file['/peaks/labels/train_out']
        flanks_train_out = h5file['/flanks/labels/train_out']
        if self.peaks_vs_flanks:
            self.train_out = np.zeros(tuple([n_peaks + n_flanks,1]), dtype=peaks_train_out.dtype)
            self.train_out[0:n_peaks] = 1
        else:
            self.train_out = np.zeros(tuple([n_peaks + n_flanks,peaks_train_out.shape[1]]), dtype=peaks_train_out.dtype)
            self.train_out[0:n_peaks,:] = peaks_train_out[:]
            self.train_out[n_peaks:n_peaks+n_flanks,:] = flanks_train_out[:]
        
        h5file.close()
    
    def load_test(self):
        h5file = self.get_h5_handle(self.data_path)
        peaks_test_in = h5file['/peaks/data/test_in']
        flanks_test_in = h5file['/flanks/data/test_in']
        n_peaks = peaks_test_in.shape[0]
        n_flanks = flanks_test_in.shape[0]
        self.test_set_size = n_peaks + n_flanks
        test_set_shape = tuple([n_peaks + n_flanks, peaks_test_in.shape[1], peaks_test_in.shape[2], peaks_test_in.shape[3]])
        self.test_in = np.zeros(test_set_shape, dtype=peaks_test_in.dtype)
        self.test_in[0:n_peaks,:,:,:] = peaks_test_in[:]
        self.test_in[n_peaks:n_peaks+n_flanks,:,:,:] = flanks_test_in[:]
        
        peaks_test_out = h5file['/peaks/labels/test_out']
        flanks_test_out = h5file['/flanks/labels/test_out']
        if self.peaks_vs_flanks:
            self.test_out = np.zeros(tuple([n_peaks + n_flanks,1]), dtype=peaks_test_in.dtype)
            self.test_out[0:n_peaks] = 1
        else:
            self.test_out = np.zeros(tuple([n_peaks + n_flanks,peaks_test_out.shape[1]]), dtype=peaks_test_out.dtype)
            self.test_out[0:n_peaks,:] = peaks_test_out[:]
            self.test_out[n_peaks:n_peaks+n_flanks,:] = flanks_test_out[:]
        
        h5file.close()
    
    def load_validation(self):
        h5file = self.get_h5_handle(self.data_path)
        peaks_valid_in = h5file['/peaks/data/valid_in']
        flanks_valid_in = h5file['/flanks/data/valid_in']
        n_peaks = peaks_valid_in.shape[0]
        n_flanks = flanks_valid_in.shape[0]
        self.valid_set_size = n_peaks + n_flanks
        valid_set_shape = tuple([n_peaks + n_flanks, peaks_valid_in.shape[1], peaks_valid_in.shape[2], peaks_valid_in.shape[3]])
        self.valid_in = np.zeros(valid_set_shape, dtype=peaks_valid_in.dtype)
        self.valid_in[0:n_peaks,:,:,:] = peaks_valid_in[:]
        self.valid_in[n_peaks:n_peaks+n_flanks,:,:,:] = flanks_valid_in[:]
        
        peaks_valid_out = h5file['/peaks/labels/valid_out']
        flanks_valid_out = h5file['/flanks/labels/valid_out']
        if self.peaks_vs_flanks:
            self.valid_out = np.zeros(tuple([n_peaks + n_flanks,1]),dtype = peaks_valid_in.dtype)
            self.valid_out[0:n_peaks] = 1
        else:
            self.valid_out = np.zeros(tuple([n_peaks + n_flanks,peaks_valid_out.shape[1]]), dtype=peaks_valid_out.dtype)
            self.valid_out[0:n_peaks,:] = peaks_valid_out[:]
            self.valid_out[n_peaks:n_peaks+n_flanks,:] = flanks_valid_out[:]
            
        h5file.close()
    
    def create_batch_gen(self, chunk_size=4096, num_chunks=58):
        if not hasattr(self, 'train_in'):
            self.load_train()
        return generators.labeled_sequence_gen(self.train_in, self.train_out, chunk_size, num_chunks)
    
    def create_buffered_gen(self, chunk_size=4096, num_chunks=58):
        if not hasattr(self, 'train_in'):
            self.load_train()
        gen = generators.labeled_sequence_gen(self.train_in, self.train_out, chunk_size, num_chunks)
        return buffering.buffered_gen_threaded(gen)   
    
    def create_valid_gen(self, chunk_size=4096, num_chunks=20):
        if not hasattr(self, 'valid_in'):
            self.load_validation()
            
        return generators.labeled_sequence_gen(self.valid_in, self.valid_out, chunk_size, num_chunks)
    
    def create_buffered_valid_gen(self, chunk_size=4096, num_chunks=12):
        if not hasattr(self, 'valid_in'):
            self.load_validation()
                    
        gen = generators.labeled_sequence_gen(self.valid_in, self.valid_out)
        return buffering.buffered_gen_threaded(gen) 

class StandardDataLoader(DataLoader):
    
    def __init__(self, **kwargs):
        DataLoader.__init__(self)
        self.__dict__.update(kwargs)
        if not hasattr(self, 'data_path'):
            self.data_path = os.path.abspath("../data/DNase/encode_roadmap_all.h5")
        
    def load_train(self,train_in_path='/train_in',train_out_path='/train_out'):
        
        h5file = self.get_h5_handle(self.data_path)
        if not hasattr(self, 'train_in_path'):
            train_in = h5file[train_in_path]
            train_out = h5file[train_out_path]
        else:
            train_in = h5file[self.train_in_path]
            train_out = h5file[self.train_out_path]
            
        self.train_set_size = train_in.shape[0]
        self.train_in = np.zeros(train_in.shape,dtype=train_in.dtype)
        self.train_in[:] = train_in[:]
        
        self.train_out = np.zeros(train_out.shape, dtype=train_out.dtype)
        self.train_out[:] = train_out[:]
        
        h5file.close()    
        
    def load_test(self,test_in_path='/test_in',test_out_path='/test_out'):
        
        h5file = self.get_h5_handle(self.data_path)
        if not hasattr(self, 'test_in_path'):
            test_in = h5file[test_in_path]
            test_out = h5file[test_out_path]
        else:
            test_in = h5file[self.test_in_path]
            test_out = h5file[self.test_out_path]
    
        self.test_in = np.zeros(test_in.shape, dtype=test_in.dtype)
        self.test_in[:] = test_in[:]
        
        self.test_out = np.zeros(test_out.shape, dtype=test_out.dtype)
        self.test_out[:] = test_out[:]
        h5file.close()  
        
    def load_validation(self,valid_in_path='/valid_in', valid_out_path='/valid_out'):
        h5file = self.get_h5_handle(self.data_path)
        if not hasattr(self, 'valid_in_path'):
            valid_in = h5file[valid_in_path]
            valid_out = h5file[valid_out_path]
        else:
            valid_in = h5file[self.valid_in_path]
            valid_out = h5file[self.valid_out_path]
            
        self.valid_set_size = valid_in.shape[0]
        self.valid_in = np.zeros(valid_in.shape, dtype=valid_in.dtype)
        self.valid_in[:] = valid_in[:]
        
        self.valid_out = np.zeros(valid_out.shape, dtype=valid_out.dtype)
        self.valid_out[:] = valid_out[:]
        h5file.close()        
        
    def create_batch_gen(self, chunk_size=4096, num_chunks=458):
        if not hasattr(self, 'train_in'):
            self.load_train()
        if hasattr(self, 'chunk_size'):
            my_chunk_size = self.chunk_size
        else:
            my_chunk_size = chunk_size
        if hasattr(self, 'num_chunks_train'):
            my_num_chunks = self.num_chunks_train
        else:
            my_num_chunks = num_chunks
    
        return generators.labeled_sequence_gen(self.train_in, self.train_out, my_chunk_size, my_num_chunks)
    
    def create_buffered_gen(self, chunk_size=4096, num_chunks=458):
        if not hasattr(self, 'train_in'):
            self.load_train()
        gen = generators.labeled_sequence_gen(self.train_in, self.train_out, chunk_size, num_chunks)
        return buffering.buffered_gen_threaded(gen)    
            
    def create_valid_gen(self, chunk_size=4096, num_chunks=17):
        if not hasattr(self, 'valid_in'):
            self.load_validation()
        if hasattr(self, 'chunk_size'):
            my_chunk_size = self.chunk_size
        else:
            my_chunk_size = chunk_size
        if hasattr(self, 'num_chunks_valid'):
            my_num_chunks = self.num_chunks_valid
        else:
            my_num_chunks = num_chunks        
            
        return generators.labeled_sequence_gen(self.valid_in, self.valid_out, my_chunk_size, my_num_chunks)
         
    def create_buffered_valid_gen(self, chunk_size, num_chunks):
        if not hasattr(self, 'valid_in'):
            self.load_validation()
                    
        gen = generators.labeled_sequence_gen(self.valid_in, self.valid_out)
        return buffering.buffered_gen_threaded(gen)    
        



class KmerDataLoader(StandardDataLoader):
    
    def __init__(self, **kwargs):
        DataLoader.__init__(self)
        self.__dict__.update(kwargs)
        if not hasattr(self, 'data_path'):
            self.data_path = os.path.abspath('../data/DNase/encode_roadmap_all.h5')
        if not hasattr(self, 'kmer_length'):
            self.kmer_length = 3
            
    def create_batch_gen(self, chunk_size=4096, num_chunks=458):
        if not hasattr(self, 'train_in'):
            self.load_train()
        if hasattr(self, 'chunk_size'):
            my_chunk_size = self.chunk_size
        else:
            my_chunk_size = chunk_size
        if hasattr(self, 'num_chunks_train'):
            my_num_chunks = self.num_chunks_train
        else:
            my_num_chunks = num_chunks
            
        return generators.labeled_kmer_sequence_gen(self.train_in, self.train_out, 
                                            self.kmer_length, my_chunk_size, my_num_chunks)
    
    def create_mismatch_batch_gen(self, chunk_size=4096, num_chunks=458):
        if not hasattr(self, 'train_in'):
            self.load_train()
        if hasattr(self, 'chunk_size'):
            my_chunk_size = self.chunk_size
        else:
            my_chunk_size = chunk_size
        if hasattr(self, 'num_chunks_train'):
            my_num_chunks = self.num_chunks_train
        else:
            my_num_chunks = num_chunks
            
        return generators.labeled_kmer_sequence_mismatch_gen(self.train_in, self.train_out, 
                                            self.kmer_length, my_chunk_size, my_num_chunks)            
    
    def create_valid_gen(self, chunk_size=4096, num_chunks=17):
        if not hasattr(self, 'valid_in'):
            self.load_validation()
        if hasattr(self, 'chunk_size'):
            my_chunk_size = self.chunk_size
        else:
            my_chunk_size = chunk_size
        if hasattr(self, 'num_chunks_valid'):
            my_num_chunks = self.num_chunks_valid
        else:
            my_num_chunks = num_chunks
            
        return generators.labeled_kmer_sequence_gen(self.valid_in, self.valid_out, 
                                            self.kmer_length, my_chunk_size, my_num_chunks)
    
    def create_mismatch_valid_gen(self, chunk_size=4096, num_chunks=17):
        if not hasattr(self, 'valid_in'):
            self.load_validation()
        if hasattr(self, 'chunk_size'):
            my_chunk_size = self.chunk_size
        else:
            my_chunk_size = chunk_size
        if hasattr(self, 'num_chunks_valid'):
            my_num_chunks = self.num_chunks_valid
        else:
            my_num_chunks = num_chunks
            
        return generators.labeled_kmer_sequence_mismatch_gen(self.valid_in, self.valid_out, 
                                            self.kmer_length, my_chunk_size, my_num_chunks)            




