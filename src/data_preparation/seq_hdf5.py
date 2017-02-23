#!/usr/bin/env python
from __future__ import print_function
from optparse import OptionParser
import sys

import h5py
import numpy.random as npr
import numpy as np

import subprocess

from utils import dna_io
import split_ttv

################################################################################
# seq_hdf5.py
#
# Make an HDF5 file for Torch input out of a FASTA file and targets text file,
# dividing the data into training, validation, and test.
#
# Basset data set is made with these args: 
# seq_hdf5.py -c -r -t 71886 -v 70000 encode_roadmap.fa encode_roadmap_act.txt encode_roadmap.h5
################################################################################

################################################################################
# main
################################################################################
def encode_sequences(my_args=None):
    usage = 'usage: %prog [options] <fasta_file> <targets_file> <out_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Align sizes with batch size')
    parser.add_option('-c', dest='counts', default=False, action='store_true', help='Validation and training proportions are given as raw counts [Default: %default]')
    parser.add_option('-e', dest='extend_length', type='int', default=None, help='Extend all sequences to this length [Default: %default]')
    parser.add_option('-r', dest='permute', default=False, action='store_true', help='Permute sequences [Default: %default]')
    parser.add_option('-s', dest='random_seed', default=1, type='int', help='numpy.random seed [Default: %default]')
    parser.add_option('-t', dest='test_pct', default=0, type='float', help='Test % [Default: %default]')
    parser.add_option('-k', dest='kmerize', default=1, type='int', help='produce kmer-ized representation of the input for this value of k')
    parser.add_option('-v', dest='valid_pct', default=0, type='float', help='Validation % [Default: %default]')
    parser.add_option('-l', dest='columns', default=600, type='int', help='number of bases (i.e feature columns) in the input')
    parser.add_option('-u', dest='chunks', default=10, type='int', help='Process the fasta file in this many chunks to conserve RAM')
    parser.add_option('-g', dest='group', default='/', type='str', help='All data (both encoded sequences and activation labels) are stored underneath this group.  Will be created if it does not arleady exist.')
    if not my_args:
        (options,args) = parser.parse_args()
    else:
        (options,args) = parser.parse_args(args=my_args)
    if len(args) != 3:
        parser.error('Must provide fasta file, targets file, and an output prefix')
    else:
        fasta_file = args[0]
        targets_file = args[1]
        out_file = args[2]

    # seed rng before shuffle
    npr.seed(options.random_seed)
    
    #################################################################
    # prepare bookeeping data for processing the sizes of each chunk
    # of the fasta input
    #################################################################    
    fasta_handle = open(fasta_file, 'r')
    targets_handle = open(targets_file, 'r')
    
        
    # get counts, train & test & validation indices + counts
    train_count, test_count, valid_count = split_ttv.split_ttv(fasta_file, options.test_pct, options.valid_pct, options.counts)
    train_indices, test_indices, validation_indices = split_ttv.split_ttv_indices(fasta_file, train_count, test_count, valid_count, options.chunks)
    ttv_counts = split_ttv.split_ttv_counts(fasta_file, train_count, test_count, valid_count, options.chunks)    

    #################################################################
    # construct hdf5 representation
    #################################################################
    h5f = h5py.File(out_file, 'a')
    target_labels = [t for t in targets_handle.readline().strip().split('\t') if not t.endswith("ID")]
    target_labels = np.array(target_labels).astype('|S21')
    
    group = h5f.create_group(options.group)
    group.create_dataset('target_labels', data=target_labels)
    data_group = group.create_group('data')
    label_group = group.create_group('labels')    
    
    alphabet_size = 4
    feature_cols = options.columns
    if options.kmerize > 1:
        alphabet_size = int(pow(4, options.kmerize))
        feature_cols = options.columns - options.kmerize + 1
    
    if train_count > 0:
        train_in_dset = data_group.create_dataset('train_in', shape=(0, alphabet_size, 1, feature_cols),  maxshape=(None, alphabet_size, 1, feature_cols), dtype='uint8')
        train_out_dset = label_group.create_dataset('train_out', shape=(0, len(target_labels)), maxshape=(None,len(target_labels)), dtype='uint8')

    if valid_count > 0:
        valid_in_dset = data_group.create_dataset('valid_in', shape=(valid_count, alphabet_size, 1, feature_cols), dtype='uint8')
        valid_out_dset = label_group.create_dataset('valid_out', shape=(valid_count, len(target_labels)), dtype='uint8')

    if test_count > 0: 
        test_in_dset = data_group.create_dataset('test_in', shape=(test_count, alphabet_size, 1, feature_cols),dtype='uint8')
        test_out_dset = label_group.create_dataset('test_out', shape=(test_count, len(target_labels)), dtype='uint8')

    #################################################################
    # load data in chunks
    #################################################################
    train_lower = 0
    test_lower = 0
    valid_lower = 0        
    
    for i, ttv_tup in enumerate(ttv_counts):
        chunksize = sum(ttv_tup)
        seqs, targets, headers = dna_io.load_data_1hot(fasta_handle, targets_handle, chunksize, extend_len=options.extend_length, mean_norm=False, whiten=False, permute=False, sort=False, kmerize=options.kmerize)
    
        print('parsed', chunksize ,' fasta sequences for batch ', i, file=sys.stderr)
        
        # reshape sequences appropriately, depending on k.  Assume that k is specified to divide 600.
        if options.kmerize > 1:
            seqs = seqs.reshape((seqs.shape[0], alphabet_size,1,seqs.shape[1] / alphabet_size))
        else:
            seqs = seqs.reshape((seqs.shape[0],4,1,seqs.shape[1]/4))
    
        if options.permute:
            order = npr.permutation(seqs.shape[0])
            seqs = seqs[order]
            targets = targets[order]
            #headers = headers[order]
    
        # check proper sum
        if options.counts:
            assert((options.test_pct/chunksize) + (options.valid_pct / chunksize) <= seqs.shape[0])
        else:
            assert(options.test_pct + options.valid_pct <= 1.0)
            
        
        #################################################################
        # divide data  
        #################################################################
        
        train_count, test_count, valid_count = ttv_tup
        
        print(seqs.shape[0], ' parsed sequences, want', sum(ttv_tup),' sequences.  Discrepancy is ', sum(ttv_tup) - seqs.shape[0], file=sys.stderr)
        print('want ', train_count,' training sequences ', file=sys.stderr)  
        print('want ', test_count,' test sequences ', file=sys.stderr)
        print('want ', valid_count, ' validation sequences', file=sys.stderr)
        
        # I'm losing sequences somewhere, so take them from the training set and resize appropriately
        discrepancy = sum(ttv_tup) - seqs.shape[0]
        train_count = train_count - discrepancy
    
        j = 0
        train_seqs, train_targets = seqs[j:j+train_count,:], targets[j:j+train_count,:]
        j += train_count
        valid_seqs, valid_targets = seqs[j:j+valid_count,:], targets[j:j+valid_count,:]
        j += valid_count
        test_seqs, test_targets = seqs[j:j+test_count,:], targets[j:j+test_count,:]
        #test_headers = headers[j:j+test_count]
        
        # calculate the upper indices for this next set of training, validation, test seqs
        train_upper = train_lower + train_count
        valid_upper = valid_lower + valid_count
        test_upper = test_lower + test_count
        
        # calculate & resize the train dataset
        new_train_in_shape = train_count + train_in_dset.shape[0]
        new_train_out_shape = train_count + train_out_dset.shape[0]
        train_in_dset.resize(new_train_in_shape, axis=0)
        train_out_dset.resize(new_train_out_shape, axis=0)
        
        train_in_dset[train_lower:train_upper,...] = train_seqs
        train_out_dset[train_lower:train_upper,...] = train_targets
        valid_in_dset[valid_lower:valid_upper,...] = valid_seqs
        valid_out_dset[valid_lower:valid_upper,...] = valid_targets
        test_in_dset[test_lower:test_upper,...] = test_seqs
        test_out_dset[test_lower:test_upper,...] = test_targets
        h5f.flush()
        
        # update the lower indices
        train_lower = train_upper
        valid_lower = valid_upper
        test_lower = test_upper


    # tidy up open files
    fasta_handle.close()
    targets_handle.close()
    h5f.close()

    

def batch_round(count, batch_size):
    if batch_size != None:
        count -= (batch_size % count)
    return count

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    encode_sequences()
