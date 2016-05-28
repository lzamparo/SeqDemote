import glob
import os

import numpy as np
import multiprocessing as mp
#import utils

DNase_train_peaks = 1880000

directories = glob.glob("../data/")
class_names = [os.path.basename(d) for d in directories]
class_names.sort()
num_classes = len(class_names)

paths_train = glob.glob("data/train/*/*")
paths_train.sort()

paths_test = glob.glob("data/test/*")
paths_test.sort()

paths = {
    'train': paths_train,
    'test': paths_test,
}


# labels_train = np.zeros(len(paths['train']), dtype='int32')
# for k, path in enumerate(paths['train']):
#     class_name = os.path.basename(os.path.dirname(path))
#     labels_train[k] = class_names.index(class_name)
labels_train = utils.load_gz("data/labels_train.npy.gz")


default_augmentation_params = {
    'kmerize': 1,
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
    'do_flip': True,
    'allow_stretch': False,
}

no_augmentation_params = {
    'zoom_range': (1.0, 1.0),
    'rotation_range': (0, 0),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
    'do_flip': False,
    'allow_stretch': False,
}

no_augmentation_params_gaussian = {
    'zoom_std': 0.0,
    'rotation_range': (0, 0),
    'shear_std': 0.0,
    'translation_std': 0.0,
    'do_flip': False,
    'stretch_std': 0.0,
}


def sequence_gen(sequences, labels, chunk_size=4096, num_chunks=458, rng=np.random):
    ''' Yield one-hot encoded sequence data one chunk at a time '''
    pass

### k-merizing generator ###

def kmerize_gen(sequences, labels, kmersize=3, chunk_size=4096, num_chunks=458, rng=np.random):
    ''' Yield positional kmer one-hot encoded data one chunk at a time '''
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
