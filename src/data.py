import os
import numpy as np

import multiprocessing as mp

### Possible data augmentation schemes by subsequence plucking, and kmerization of different varieties

default_augmentation_params = {
    'kmerize': 1,
    'subsequence_range': (-500, 500),
    'subsequence_length': 500,
    'do_subsequences': False,
}

kmer_augmentation_params = {
    'kmerize': 4,
    'subsequence_range': (-300, 300),
    'subsequence_length': 0,
    'do_subsequences': False,
}

subsequence_augmentation_params = {
    'kmerize': 1,
    'subsequence_range': (-500, 500),
    'subsequence_length': 500,
    'do_subsequences': False,
}


### Data provider generators

def get_h5_handle(path):
    """
    Get the h5 handle to the
    """
    return h5py.File(path)


def train_sequence_gen(sequences, labels, chunk_size=4096, num_chunks=458, rng=np.random):
    ''' Given training data array ref, build and return a generator for one-hot encoded training sequence data '''
    for n in xrange(num_chunks):
        indices = rng.randint(0, len(sequences), chunk_size)
        
        sequences_rows = sequences.shape[2]
        sequences_cols = sequences.shape[3]
        
        chunk_x = np.zeros((chunk_size, sequences_rows, sequences_cols), dtype='float32')
        chunk_y = np.zeros((chunk_size,), dtype='float32')
        
        for k, idx in enumerate(indices):
            chunk_x[k] = sequences[indices[k]]
            chunk_y[k] = labels[indices[k]]
            
        yield chunk_x, chunk_y
    

def train_kmerize_gen(sequences, labels, kmersize=3, chunk_size=4096, num_chunks=458, rng=np.random):
    ''' Given training data array ref, build and return a generator for one-hot encoded kmerized training sequence data '''
    pass


####  augmentation  ####
#  None of the regular image augmentation methods will work for me, but what might work is to re-process the data with a much larger window set
#  and then take random 600bp (or whatever window size) around the center of the peak.

def subseq_gen(sequences, labels, subseq_size = (600), chunk_size=4096, num_chunks=458, rng=np.random):
    ''' Yield a random subsequence of window size subseq_size one chunk at a time'''
    pass


##### legacy code from Sander Dieleman's plankton kaggle code

#def patches_gen(images, labels, patch_size=(50, 50), chunk_size=4096, num_chunks=100, rng=np.random):
    #p_x, p_y = patch_size

    #for n in xrange(num_chunks):
        #indices = rng.randint(0, len(images), chunk_size)

        #chunk_x = np.zeros((chunk_size, p_x, p_y), dtype='float32')
        #chunk_y = np.zeros((chunk_size,), dtype='float32')

        #for k, idx in enumerate(indices):
            #img = images[indices[k]]
            #extract_image_patch(chunk_x[k], img)
            #chunk_y[k] = labels[indices[k]]
        
        #yield chunk_x, chunk_y


#def patches_gen_ordered(images, patch_size=(50, 50), chunk_size=4096):
    #p_x, p_y = patch_size

    #num_images = len(images)
    #num_chunks = int(np.ceil(num_images / float(chunk_size)))

    #idx = 0

    #for n in xrange(num_chunks):
        #chunk_x = np.zeros((chunk_size, p_x, p_y), dtype='float32')
        #chunk_length = chunk_size

        #for k in xrange(chunk_size):
            #if idx >= num_images:
                #chunk_length = k
                #break

            #img = images[idx]
            #extract_image_patch(chunk_x[k], img)
            #idx += 1

        #yield chunk_x, chunk_length


#def patches_gen_augmented(images, labels, patch_size=(50, 50), chunk_size=4096,
        #num_chunks=100, rng=np.random, rng_aug=np.random, augmentation_params=default_augmentation_params):
    #p_x, p_y = patch_size

    #if augmentation_params is None:
        #augmentation_params = no_augmentation_params

    #for n in xrange(num_chunks):
        #indices = rng.randint(0, len(images), chunk_size)

        #chunk_x = np.zeros((chunk_size, p_x, p_y), dtype='float32')
        #chunk_y = labels[indices].astype('float32')

        #for k, idx in enumerate(indices):
            #img = images[idx]
            #img = uint_to_float(img)
            #chunk_x[k] = perturb(img, augmentation_params, target_shape=patch_size, rng=rng_aug)
        
        #yield chunk_x, chunk_y
