from __future__ import print_function

from nose.tools import eq_, ok_

import nose, functools
import os
import numpy as np

import load
from utils import dna_io

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

### Kmer fixtures

aaa_3mer_string = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
aaa_3mer_position = 0

ttt_3mer_string = 'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT'
ttt_3mer_position = 63

aaa_4mer_string = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
ttt_4mer_string = 'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT'

rando_3mer_string = 'ATGGGGTAGAGAATGGGGTAGAGACCAGGT'

alphabet_size_3 = 64
alphabet_size_4 = 256


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

def test_encoding_kmerizing_threemers():
    
    # encode the fixtures as a one-hot encoded matrix
    my_seqs = [aaa_3mer_string,ttt_3mer_string,aaa_3mer_string,ttt_3mer_string]    
    seqs = np.vstack([dna_io.dna_one_hot(seq) for seq in my_seqs])
    
    # reshape into expected shape
    seqs = seqs.reshape((seqs.shape[0],4,1,seqs.shape[1]/4))
    
    # re-encode the one-hot examples as positional 3-mers
    seqs_3mers = dna_io.one_hot_to_kmerized(seqs, 3)
    seqs_3mers_reshaped = seqs_3mers.reshape((seqs_3mers.shape[0],alphabet_size_3,1,len(aaa_3mer_string)-2))
    
    # make up the positional 3-mer fixture
    aaa_vec = dna_io.dna_one_hot_kmer(aaa_3mer_string, 3)
    aaa_mat = aaa_vec.reshape((1,alphabet_size_3,1,len(aaa_3mer_string) - 2))
    ttt_vec = dna_io.dna_one_hot_kmer(ttt_3mer_string, 3)
    ttt_mat = ttt_vec.reshape((1,alphabet_size_3,1,len(ttt_3mer_string) - 2))
    
    fixture = np.vstack((aaa_mat,ttt_mat,aaa_mat,ttt_mat))
    
    for fix, elem in zip(fixture,seqs_3mers_reshaped):
        ok_(np.allclose(fix, elem))
        
def test_decoding_kmerizing_threemers():
    # encode the fixtures as a one-hot encoded matrix
    my_seqs = [aaa_3mer_string,ttt_3mer_string,aaa_3mer_string,ttt_3mer_string]    
    seqs = np.vstack([dna_io.dna_one_hot(seq) for seq in my_seqs])
    
    # reshape into expected shape
    seqs = seqs.reshape((seqs.shape[0],4,1,seqs.shape[1]/4))
    
    # re-encode the one-hot examples as positional 3-mers
    seqs_3mers = dna_io.one_hot_to_kmerized(seqs, 3)
    seqs_3mers_reshaped = seqs_3mers.reshape((seqs_3mers.shape[0],alphabet_size_3,1,len(aaa_3mer_string)-2))  
    
    # de-code the encoded 3-mers
    decoded_seqs = dna_io.kmer_vecs_to_dna(seqs_3mers_reshaped, 3)
    for seq, fix in zip(decoded_seqs,my_seqs):
        eq_(seq,fix)
        
def test_encoding_kmerizing_fourmers():
    
    # encode the fixtures as one-hot encoded matrix
    my_seqs = [aaa_4mer_string,ttt_4mer_string, aaa_4mer_string, ttt_4mer_string]
    seqs = np.vstack([dna_io.dna_one_hot(seq) for seq in my_seqs])
    
    # reshape into expected shape
    seqs = seqs.reshape((seqs.shape[0], 4, 1, seqs.shape[1]/4))
    
    # re-encode the one-hot examles as positional 4-mers
    seqs_4mers = dna_io.one_hot_to_kmerized(seqs, 4)
    seqs_4mers_reshaped = seqs_4mers.reshape((seqs_4mers.shape[0],alphabet_size_4,1,len(aaa_4mer_string)-3))
    
    # make up the positional 4-mer fixture
    aaa_vec = dna_io.dna_one_hot_kmer(aaa_4mer_string,4)
    aaa_mat = aaa_vec.reshape((1,alphabet_size_4,1,len(aaa_4mer_string)-3))
    ttt_vec = dna_io.dna_one_hot_kmer(ttt_4mer_string, 4)
    ttt_mat = ttt_vec.reshape((1,alphabet_size_4,1,len(ttt_4mer_string)-3))
    
    fixture = np.vstack((aaa_mat,ttt_mat,aaa_mat,ttt_mat))
    
    for fix, elem in zip(fixture, seqs_4mers_reshaped):
        ok_(np.allclose(fix,elem))
    

def test_decoding_kmerizing_fourmers():
    
    # encode the fixtures as one-hot encoded matrix
    my_seqs = [aaa_4mer_string, ttt_4mer_string, aaa_4mer_string, ttt_4mer_string]
    seqs = np.vstack([dna_io.dna_one_hot(seq) for seq in my_seqs])
    
    # reshape into expected shape
    seqs = seqs.reshape((seqs.shape[0], 4, 1, seqs.shape[1]/4))
    
    # re-encode the one-hot examples as positional 4-mers
    seqs_4mers = dna_io.one_hot_to_kmerized(seqs,4)
    seqs_4mers_reshaped = seqs_4mers.reshape((seqs_4mers.shape[0], alphabet_size_4, 1, len(aaa_4mer_string)-3))
    
    # decode the encoded 4-mers
    decoded_seqs = dna_io.kmer_vecs_to_dna(seqs_4mers_reshaped, 4)
    for seq, fix in zip(decoded_seqs,my_seqs):
        eq_(seq,fix)
    
if __name__ == "__main__":
    #test_build_data_loader()
    #test_build_data_loader_kwargs()
    #test_dnase_data_shape()
    #test_exhaust_data()
    #test_training_batch_encoding_sum()
    #test_validation_batch_encoding_sum()
    #test_encoding_kmerizing_threemers()
    #test_decoding_kmerizing_threemers()
    test_encoding_kmerizing_fourmers()
    test_decoding_kmerizing_fourmers()