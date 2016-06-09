import sys
import numpy as np
import itertools

import generators
import os
import buffering
import h5py

class DataLoader(object):
    params = {} # attributes that need to be stored after training and loaded at test time.

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        #if not hasattr(self, 'validation_split_path'):
            #self.validation_split_path = DEFAULT_VALIDATION_SPLIT_PATH
            #print "using default validation split: %s" % self.validation_split_path
        #else:
            #print "using NON-default validation split: %s" % self.validation_split_path

    def load_train(self):   
        h5file = self.get_h5_handle(self.data_path)
        train_in = h5file['/train_in']
        self.train_set_size = train_in.shape[0]
        train_out = h5file['/train_out']
        self.train_in = np.zeros(train_in.shape,dtype=train_in.dtype)
        self.train_in[:] = train_in[:]
        
        self.train_out = np.zeros(train_out.shape, dtype=train_out.dtype)
        self.train_out[:] = train_out[:]
        
        h5file.close()

    def load_test(self):
        h5file = self.get_h5_handle(self.data_path)
        test_in = h5file['/test_in']
        test_out = h5file['test_out']
        
        self.test_in = np.zeros(test_in.shape, dtype=test_in.dtype)
        self.test_in[:] = test_in[:]
        
        self.test_out = np.zeros(test_out.shape, dtype=test_out.dtype)
        self.test_out[:] = test_out[:]
        h5file.close()
        
    def load_validation(self):
        h5file = self.get_h5_handle(self.data_path)
        valid_in = h5file['/valid_in']
        valid_out = h5file['valid_out']
        self.valid_set_size = valid_in.shape[0]
        
        self.valid_in = np.zeros(valid_in.shape, dtype=valid_in.dtype)
        self.valid_in[:] = valid_in[:]
        self.valid_out = np.zeros(valid_out.shape, dtype=valid_out.dtype)
        self.valid_out[:] = valid_out[:]
        h5file.close()

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


class DNaseDataLoader(DataLoader):
    
    def __init__(self, **kwargs):
        DataLoader.__init__(self)
        self.__dict__.update(kwargs)
        if not hasattr(self, 'data_path'):
            self.data_path = os.path.abspath("../data/DNase/encode_roadmap.h5")
        
    
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
    
        return generators.train_sequence_gen(self.train_in, self.train_out, my_chunk_size, my_num_chunks)
    
    def create_buffered_gen(self, chunk_size=4096, num_chunks=458):
        if not hasattr(self, 'train_in'):
            self.load_train()
        gen = generators.train_sequence_gen(self.train_in, self.train_out, chunk_size, num_chunks)
        return buffering.buffered_gen_threaded(gen)    
            
    def create_valid_gen(self, chunk_size, num_chunks):
        if not hasattr(self, 'train_in'):
            self.load_valid()
        if hasattr(self, 'chunk_size'):
            my_chunk_size = self.chunk_size
        else:
            my_chunk_size = chunk_size
        if hasattr(self, 'num_chunks_valid'):
            my_num_chunks = self.num_chunks_valid
        else:
            my_num_chunks = num_chunks        
            
        return generators.train_sequence_gen(self.valid_in, self.valid_out, my_chunk_size, my_num_chunks)
         
    def create_buffered_valid_gen(self, chunk_size, num_chunks):
        if not hasattr(self, 'train_in'):
            self.load_valid()
                    
        gen = generators.train_sequence_gen(self.valid_in, self.valid_out)
        return buffering.buffered_gen_threaded(gen)    
        
class RescaledDataLoader(DataLoader):
    def create_random_gen(self, images, labels):
        gen = generators.rescaled_patches_gen_augmented(images, labels, self.estimate_scale, patch_size=self.patch_size,
            chunk_size=self.chunk_size, num_chunks=self.num_chunks_train, augmentation_params=self.augmentation_params)

        def random_gen():
            for chunk_x, chunk_y, chunk_shape in gen:
                yield [chunk_x[:, None, :, :]], chunk_y

        return buffering.buffered_gen_threaded(random_gen())

    def create_fixed_gen(self, images, augment=False):
        augmentation_transforms = self.augmentation_transforms_test if augment else None
        gen = generators.rescaled_patches_gen_fixed(images, self.estimate_scale, patch_size=self.patch_size,
            chunk_size=self.chunk_size, augmentation_transforms=augmentation_transforms)
        
        def fixed_gen():
            for chunk_x, chunk_shape, chunk_length in gen:
                yield [chunk_x[:, None, :, :]], chunk_length

        return buffering.buffered_gen_threaded(fixed_gen())


