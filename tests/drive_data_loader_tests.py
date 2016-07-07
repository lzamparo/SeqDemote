from __future__ import print_function

from nose.tools import eq_, ok_

import nose, functools
import os
import numpy as np

import load

def expected_failure(test):
    @functools.wraps(test)
    def inner(*args, **kwargs):
        try:
            test(*args, **kwargs)
        except Exception:
            raise nose.SkipTest
        else:
            raise AssertionError('Failure expected')
    return inner

### DNase data fixtures
train_size = 1880000
valid_size = 70000
output_size = 164
chunk_size = 4096
batch_size = 128
num_chunks_train = train_size // chunk_size

full_train_shape = (train_size, 4, 1, 600)
chunk_train_shape = (chunk_size,4,1,600)
chunk_out_shape = (chunk_size,output_size)
batch_train_shape = (batch_size,4,1,600)

path = os.path.expanduser("~/projects/SeqDemote/data/DNase/encode_roadmap.h5")

def test_build_data_loader():
    """ Can I build a data loader for the DNase data """
    
    data_loader = load.DNaseDataLoader(chunk_size=chunk_size, batch_size=batch_size, num_chunks_train=num_chunks_train)
    data_loader.load_train()
    eq_(data_loader.train_in.shape, full_train_shape)
    
def test_build_data_loader_kwargs():
    """ Can I build a data loader for the DNase data specifying the data path """
    
    data_loader = load.DNaseDataLoader(data_path=path,chunk_size=chunk_size, batch_size=batch_size, num_chunks_train=num_chunks_train)
    data_loader.load_train()
    eq_(data_loader.train_in.shape, full_train_shape)
    eq_(data_loader.chunk_size, chunk_size)
    eq_(data_loader.num_chunks_train, num_chunks_train)

def test_dnase_data_shape():
    """ Is my DNase data the right size and shape """

    data_loader = load.DNaseDataLoader(data_path=path)
    data_loader.load_train()
    num_chunks = range(num_chunks_train)
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_batch_gen()):
        print("Chunk: ", str(e + 1), " of ", num_chunks_train)
        eq_(x_chunk.shape, chunk_train_shape)
        eq_(y_chunk.shape, chunk_out_shape)
    
def test_exhaust_data():
    """ If I iterate through all the chunks, how many data points do I see? """
    seen_pts = 0
    
    data_loader = load.DNaseDataLoader(data_path=path)
    data_loader.load_train()
    num_chunks = range(num_chunks_train)
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_batch_gen()):
        seen_pts = seen_pts + x_chunk.shape[0]
    print("Saw ", str(seen_pts), " points total")    
    ok_(seen_pts <= train_size)
    
### Semantic tests: are all examples properly one-hot encoded?
    
def test_training_batch_encoding_sum():
    """ If I sum all elements of a chunk of training data, do I get the number of expected ones? """
    
    data_loader = load.DNaseDataLoader(data_path=path)
    data_loader.load_train()
    num_chunks = range(num_chunks_train)
    expected_chunk_sum = 600 * chunk_size
    ### each chunk of should have |seq_length| * |batch_size| * |chunk_size| number of 1s
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_batch_gen()):
        ones_per_chunk = np.sum(x_chunk)
        #print("Chunk  ", str(e + 1), " has ", ones_per_chunk, " bits turned on")
        eq_(ones_per_chunk, expected_chunk_sum)
        
def test_validation_batch_encoding_sum():
    """ If I sum all elements of a chunk of validation data, do I get the number of expected ones? """
    
    data_loader = load.DNaseDataLoader(data_path=path)
    data_loader.load_validation()
    num_chunks = range(num_chunks_train)
    expected_chunk_sum = 600 * chunk_size
    ### each chunk of should have |seq_length| * |batch_size| * |chunk_size| number of 1s
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_valid_gen()):
        ones_per_chunk = np.sum(x_chunk)
        #print("Chunk  ", str(e + 1), " has ", ones_per_chunk, " bits turned on")
        eq_(ones_per_chunk, expected_chunk_sum)        
        
        
def test_any_always_negatives_training():
    """ Are there any always-negative examples in the training set? """
    
    data_loader = load.DNaseDataLoader(data_path=path)
    data_loader.load_train()    
    for e, (x_chunk, y_chunk) in zip(range(num_chunks_train),data_loader.create_batch_gen()):
        for label_vector in y_chunk:
            total_peaks_on = np.sum(label_vector)
            ok_(total_peaks_on > 0)    
            
def test_any_always_negatives_validation():
    """ Are there any always-negative examples in the validation set? """
    
    data_loader = load.DNaseDataLoader(data_path=path)
    data_loader.load_validation()    
    for e, (x_chunk, y_chunk) in zip(range(num_chunks_valid),data_loader.create_valid_gen()):
        for label_vector in y_chunk:
            total_peaks_on = np.sum(label_vector)
            ok_(total_peaks_on > 0)
    
if __name__ == "__main__":
    #test_build_data_loader()
    #test_build_data_loader_kwargs()
    #test_dnase_data_shape()
    #test_exhaust_data()
    test_training_batch_encoding_sum()
    test_validation_batch_encoding_sum()
    