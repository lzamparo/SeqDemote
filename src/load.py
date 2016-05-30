import numpy as np
import itertools

import data
import os
import buffering
import h5py

class DataLoader(object):
    params = [] # attributes that need to be stored after training and loaded at test time.

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        #if not hasattr(self, 'validation_split_path'):
            #self.validation_split_path = DEFAULT_VALIDATION_SPLIT_PATH
            #print "using default validation split: %s" % self.validation_split_path
        #else:
            #print "using NON-default validation split: %s" % self.validation_split_path

    def load_train(self):   
        h5file = data.get_h5_handle(self.data_path)
        train_in = h5file['/train_in']
        train_out = h5file['/train_out']
        self.train_in = train_in
        self.train_out = train_out
        h5file.close()

    def load_test(self):
        h5file = data.get_h5_handle(self.data_path)
        test_in = h5file['/test_in']
        test_out = h5file['test_out']
        self.test_in = test_in
        self.test_out = test_out
        h5file.close()
        
    def load_validation(self):
        h5file = data.get_h5_handle(self.data_path)
        valid_in = h5file['/valid_in']
        valid_out = h5file['valid_out']
        self.valid_in = valid_in
        self.valid_out = valid_out
        h5file.close()

    def get_params(self):
        return { pname: getattr(self, pname, None) for pname in self.params }
        
    def set_params(self, p):
        self.__dict__.update(p)

        
class DNaseDataLoader(DataLoader):
    
    def __init__(self, **kwargs):
        DataLoader.__init__(self)
        self.__dict__.update(kwargs)
        self.data_path = os.path.abspath("../data/DNase/encode_roadmap.h5")
        self.train_set_size = 1880000        
    
    def create_batch_gen(self, sequences, labels, chunk_size=4096, num_chunks=458):
        gen = data.train_sequence_gen(sequences, labels, chunk_size, num_chunks, 
                                     rng=np.random)
        return buffering.buffered_gen_threaded(gen)
            
            
        
class RescaledDataLoader(DataLoader):
    def create_random_gen(self, images, labels):
        gen = data.rescaled_patches_gen_augmented(images, labels, self.estimate_scale, patch_size=self.patch_size,
            chunk_size=self.chunk_size, num_chunks=self.num_chunks_train, augmentation_params=self.augmentation_params)

        def random_gen():
            for chunk_x, chunk_y, chunk_shape in gen:
                yield [chunk_x[:, None, :, :]], chunk_y

        return buffering.buffered_gen_threaded(random_gen())

    def create_fixed_gen(self, images, augment=False):
        augmentation_transforms = self.augmentation_transforms_test if augment else None
        gen = data.rescaled_patches_gen_fixed(images, self.estimate_scale, patch_size=self.patch_size,
            chunk_size=self.chunk_size, augmentation_transforms=augmentation_transforms)
        
        def fixed_gen():
            for chunk_x, chunk_shape, chunk_length in gen:
                yield [chunk_x[:, None, :, :]], chunk_length

        return buffering.buffered_gen_threaded(fixed_gen())


