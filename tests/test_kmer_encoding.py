from __future__ import print_function
import data_load_utils as utils

from nose.tools import eq_
import nose, functools

import dna_io
import numpy as np

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


''' fixtures '''

aaa_3mer_string = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
aaa_3mer_position = 0

ttt_3mer_string = 'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT'
ttt_3mer_position = 21

rando_3mer_string = 'ATGGGGTAGAGAATGGGGTAGAGA'

alphabet_size_3 = 64
alphabet_size_4 = 256

def test_aaa_encoding():
    aaa_vec = dna_io.dna_one_hot_kmer(aaa_3mer_string, 3)
    aaa_mat = aaa_vec.reshape((alphabet_size_3,len(aaa_3mer_string) - 2))
    nonzeros = aaa_mat.nonzero()[0]
    fixture = np.ones_like(nonzeros)
    fixture = fixture * aaa_3mer_position
    np.testing.assert_array_equal(fixture, nonzeros)
    

def test_ttt_encoding():
    ttt_vec = dna_io.dna_one_hot_kmer(ttt_3mer_string, 3)
    ttt_mat = ttt_vec.reshape((alphabet_size_3,len(ttt_3mer_string) - 2))
    nonzeros = ttt_mat.nonzero()[0]
    fixture = np.ones_like(nonzeros)
    fixture = fixture * ttt_3mer_position    
    np.testing.assert_array_equal(fixture, nonzeros)
    

def test_aaa_decoding():
    aaa_vec = dna_io.dna_one_hot_kmer(aaa_3mer_string, 3)
    result = dna_io.kmer_vecs_to_dna(aaa_vec,3)
    eq_(result[0],aaa_3mer_string)
    

def test_ttt_decoding():
    ttt_vec = dna_io.dna_one_hot_kmer(ttt_3mer_string, 3)
    result = dna_io.kmer_vecs_to_dna(ttt_vec,3)
    eq_(result[0],ttt_3mer_string)
    
def test_rando_decoding():
    rando_vec = dna_io.dna_one_hot_kmer(rando_3mer_string, 3)
    result = dna_io.kmer_vecs_to_dna(rando_vec,3)
    eq_(result[0], rando_3mer_string)


def test_encoding_shape_from_file():
    # choose a random test file
    test_files = utils.get_test_data_files()
    arr = np.arange(len(test_files))
    np.random.shuffle(arr)
    test_file = test_files[arr[0]]
    test_seqs = utils.load_data_from_file(test_file)
    #DEBUG [print("sequence length: ",len(s)) for s in test_seqs]  
        
    # encode
    encoded_test_seqs = []
    for seq in test_seqs:
        encoded_test_seqs.append(dna_io.dna_one_hot_kmer(seq,3))
        
    # stack into matrices, check the shapes match
    train_seqs = np.vstack(encoded_test_seqs)
    desired_shape = (len(test_seqs),alphabet_size_3 * (len(test_seqs[0]) - 2))
    assert((desired_shape[0] == train_seqs.shape[0]) and (desired_shape[1] == train_seqs.shape[1]))


def test_end_to_end_3mer_good_data():
    # choose the caps_only.fa file
    test_files = utils.get_test_data_files()
    caps_only = [f for f in test_files if f.endswith('caps_only.fa')]
    test_seqs = utils.load_data_from_file(caps_only[0],trunc=9)
    
    # encode
    encoded_test_seqs = []
    for seq in test_seqs:
        encoded_test_seqs.append(dna_io.dna_one_hot_kmer(seq,3))
    
    # stack into matrices
    train_seqs = np.vstack(encoded_test_seqs)    
    
    # decode
    decoded_test_seqs = dna_io.kmer_vecs_to_dna(train_seqs, 3)
    
    # compare elementwise
    agreement = True
    for enc, dec in zip(test_seqs,decoded_test_seqs):
        agreement = agreement and (enc == dec)
        if not agreement:
            print("Expected ", enc, " ||||||||| but got ", dec)
    
    assert agreement

@expected_failure    
def test_end_to_end_baddata():
    # choose the caps_only.fa file
    test_files = utils.get_test_data_files()
    caps_only = [f for f in test_files if f == 'first_50.fa']
    test_seqs = utils.load_data_from_file(caps_only,trunc=9)
    
    # encode
    encoded_test_seqs = []
    for seq in test_seqs:
        encoded_test_seqs.append(dna_io.dna_one_hot_kmer(seq,3))
    
    # stack into matrices
    train_seqs = np.vstack(encoded_test_seqs)    
    
    # decode
    decoded_test_seqs = dna_io.kmer_vecs_to_dna(train_seqs, 3)
    
    # compare elementwise
    agreement = True
    for enc, dec in zip(test_seqs,decoded_test_seqs):
        agreement = agreement and (enc == dec)
        if not agreement:
            print("Expected ", enc, " ||||||||| but got ", dec)
    
    assert agreement
    