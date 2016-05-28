import numpy as np
import itertools

import data
import os
#import utils
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

    def estimate_params(self):
        pass

    def load_train(self):
        #images = data.load('train')
        #labels = utils.one_hot(data.labels_train, m=121).astype(np.float32)

        #split = np.load(self.validation_split_path)
        #indices_train = split['indices_train']
        #indices_valid = split['indices_valid']

        #self.images_train = images[indices_train]
        #self.labels_train = labels[indices_train]
        #self.images_valid = images[indices_valid]
        #self.labels_valid = labels[indices_valid]
        
        pass

    def load_test(self):
        pass
        
    def load_validation(self):
        pass

    def get_params(self):
        return { pname: getattr(self, pname, None) for pname in self.params }
        
    def set_params(self, p):
        self.__dict__.update(p)




### DNase data looks like this:
        #test_in                  Dataset {71886, 4, 1, 600}
        #test_out                 Dataset {71886, 164}
        #train_in                 Dataset {1880000, 4, 1, 600}
        #train_out                Dataset {1880000, 164}
        #valid_in                 Dataset {70000, 4, 1, 600}
        #valid_out                Dataset {70000, 164}        
        
class DNaseDataLoader(DataLoader):
    def get_h5_handle():
        """
        Load all sequences into memory for faster processing
        """
        path = os.path.abspath("../data/DNase/encode_roadmap.h5")
        return h5py.File(path)
   
    def load_train(self):
        data_file = self.get_h5_handle()
        train_in = data_file['/train_in']
        train_out = data_file['/train_out']
        data_file.close()
        return (train_in, train_out)
    
    def load_test(self):
        pass
    
    def load_validation(self):
        pass
    
    def create_batch_gen(self, sequences, labels):
        pass
            
    
        
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


