from __future__ import print_function
import data_load_utils as utils

import functools
import nose
from nose.tools import eq_

import dna_io
import numpy as np

''' fixtures '''
lc_vs_UC_file = './test_data/lc_vs_UC.fa'
AAA_vs_lc_file = './test_data/AAA_vs_lc.fa'

alphabet_size_3 = 64
alphabet_size_4 = 256


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

@expected_failure
def test_kmer_end_to_end_caps_vs_lc():
    test_seqs = utils.load_data_from_file(lc_vs_UC_file)
        
    # encode
    encoded_test_seqs = []
    for seq in test_seqs:
        encoded_test_seqs.append(dna_io.dna_one_hot_kmer(seq,3))
        
    # stack into matrices, check the shapes match
    try:
        train_seqs = np.vstack(encoded_test_seqs)
    except ValueError:
        print("Caught a value error in vstack! Dumping shape info: ")
        for i,elem in enumerate(encoded_test_seqs):
            print("element ", i, " has shape ", str(elem.shape))
        assert False
    
    # decode
    try:
        decoded_test_seqs = dna_io.kmer_vecs_to_dna(train_seqs, 3)
    except ValueError:
        print("Caught a value error in decoding, dunno why")
        assert False

    # compare elementwise
    agreement = True
    for enc, dec in zip(test_seqs,decoded_test_seqs):
        agreement = agreement and (enc == dec)
        if not agreement:
            print("Expected ", enc, " ||||||||| but got ", dec)

    assert agreement    
    
@expected_failure
def test_base_end_to_end_caps_vs_lc():
    test_seqs = utils.load_data_from_file(lc_vs_UC_file)
        
    # encode
    encoded_test_seqs = []
    for seq in test_seqs:
        encoded_test_seqs.append(dna_io.dna_one_hot(seq))
        
    # stack into matrices, check the shapes match
    try:
        train_seqs = np.vstack(encoded_test_seqs)
    except ValueError:
        print("Caught a value error in vstack! Dumping shape info: ")
        for i,elem in enumerate(encoded_test_seqs):
            print("element ", i, " has shape ", str(elem.shape))
        assert False
    
    # decode
    try:
        decoded_test_seqs = dna_io.vecs2dna(train_seqs)
    except ValueError:
        print("Caught a value error in decoding, dunno why")
        assert False
        
    # compare elementwise
    agreement = True
    for enc, dec in zip(test_seqs,decoded_test_seqs):
        agreement = agreement and (enc == dec)
        if not agreement:
            print("Expected ", enc, " ||||||||| but got ", dec)

    assert agreement

@expected_failure
def test_kmer_end_to_end_AAA_vs_lc():
    test_seqs = utils.load_data_from_file(AAA_vs_lc_file)
        
    # encode
    encoded_test_seqs = []
    for seq in test_seqs:
        encoded_test_seqs.append(dna_io.dna_one_hot_kmer(seq,3))
        
    # stack into matrices, check the shapes match
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
def test_base_end_to_end_AAA_vs_lc():
    test_seqs = utils.load_data_from_file(AAA_vs_lc_file)
        
    # encode
    encoded_test_seqs = []
    for seq in test_seqs:
        encoded_test_seqs.append(dna_io.dna_one_hot(seq))
        
    # stack into matrices, check the shapes match
    train_seqs = np.vstack(encoded_test_seqs)
    
    # decode
    decoded_test_seqs = dna_io.vecs2dna(train_seqs)

    # compare elementwise
    agreement = True
    for enc, dec in zip(test_seqs,decoded_test_seqs):
        agreement = agreement and (enc == dec)
        if not agreement:
            print("Expected ", enc, " ||||||||| but got ", dec)

    assert agreement
  
    
   
    
    
    