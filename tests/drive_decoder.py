from __future__ import print_function
import data_load_utils as utils

from nose.tools import eq_

import dna_io
import numpy as np
import khmer

rando_3mer_string = 'ATGGGGTAGAGAATGGGGTAGAGA'
AAA_vs_lc_file = './test_data/AAA_vs_lc.fa'
lc_vs_UC_file = './test_data/lc_vs_UC.fa'

    
def test_rando_decoding():
    ktable = khmer.new_ktable(3)
    rando_vec = dna_io.dna_one_hot_kmer(rando_3mer_string, 3, ktable)
    kmer_decoder = {i: ktable.reverse_hash(i) for i in range(0, ktable.n_entries())}
    result = dna_io.kmer_vecs_to_dna(rando_vec,3, kmer_decoder)
    print("expected: ", rando_3mer_string, " and got: ", result[0])


def test_end_to_end_3mer():
    # choose a random test file
    test_files = utils.get_test_data_files()
    arr = np.arange(len(test_files))
    np.random.shuffle(arr)
    test_file = test_files[arr[0]]
    test_seqs = utils.load_data_from_file(test_file,trunc=9)
    
    # encode
    encoded_test_seqs = []
    for seq in test_seqs:
        encoded_test_seqs.append(dna_io.dna_one_hot_kmer(seq,3))
    
    # stack into matrices
    train_seqs = np.vstack(encoded_test_seqs)    
    
    # decode
    decoded_test_seqs = dna_io.kmer_vecs_to_dna(train_seqs, 3)
    print("length of decoded test seqs is ", len(decoded_test_seqs))
    print("length of first element is ", len(decoded_test_seqs[0]))
    print("first decoded seq: ", decoded_test_seqs[0])
    print("first test seq equals first decoded seq?: ", str(test_seqs[0] == decoded_test_seqs[0]))
    
    # compare elementwise
    agreement = True
    for enc, dec in zip(test_seqs,decoded_test_seqs):
        agreement = agreement and (enc == dec)
        if not agreement:
            print("Expected ", enc, " ||||||||| but got ", dec)
    
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
    
if __name__ == "__main__":
    #test_base_end_to_end_AAA_vs_lc()
    test_kmer_end_to_end_AAA_vs_lc()
    #test_base_end_to_end_caps_vs_lc()
    #test_kmer_end_to_end_caps_vs_lc()