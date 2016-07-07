#!/usr/bin/env python
import sys
from collections import OrderedDict

import itertools

import numpy as np
import numpy.random as npr
from sklearn import preprocessing

################################################################################
# dna_io.py
#
# Methods to load the training data.
################################################################################

################################################################################
# align_seqs_scores
#
# Align entries from input dicts into numpy matrices ready for analysis.
#
# Input
#  seq_vecs:      Dict mapping headers to sequence vectors.
#  seq_scores:    Dict mapping headers to score vectors.
#
# Output
#  train_seqs:    Matrix with sequence vector rows.
#  train_scores:  Matrix with score vector rows.
################################################################################
def align_seqs_scores_1hot(seq_vecs, seq_scores, sort=True):
    if sort:
        seq_headers = sorted(seq_vecs.keys())
    else:
        seq_headers = seq_vecs.keys()

    # construct lists of vectors
    train_scores = []
    train_seqs = []
    for header in seq_headers:
        train_seqs.append(seq_vecs[header])
        train_scores.append(seq_scores[header])

    # stack into matrices
    train_seqs = np.vstack(train_seqs)
    train_scores = np.vstack(train_scores)

    return train_seqs, train_scores


################################################################################
# dna_one_hot
#
# Input
#  seq:
#
# Output
#  seq_vec: Flattened column vector
################################################################################

def dna_one_hot(seq, seq_len=None, flatten=True):
    if seq_len == None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = int((len(seq)-seq_len)/2)
            seq = seq[seq_trim:seq_trim+seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len-len(seq))/2
            
    ### BUG BUG BUG, this will not encode 'acgt', which is bogus as all hell.
    #seq = seq.replace('A','0')
    #seq = seq.replace('C','1')
    #seq = seq.replace('G','2')
    #seq = seq.replace('T','3')
    
    decoder = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}

    # map nt's to a matrix 4 x len(seq) of 0's and 1's.
    seq_code = np.zeros((4,seq_len), dtype='int8')
    mistakes = 0
    for i in range(seq_len):
        if i < seq_start:
            seq_code[:,i] = 0.25
        else:
            try:
                seq_code[decoder[seq[i-seq_start]],i] = 1
            except:
                # this fails with dype='int8'
                # change to 'float16' and test it
                # seq_code[:,i] = 0.25
                mistakes = mistakes + 1
                pass

    if mistakes > 0:
        print("DEBUG: made ", " mistakes for this sequence: ", seq)
    # flatten and make a column vector 1 x len(seq)
    if flatten:
        seq_vec = seq_code.flatten()[None,:]

    return seq_vec




################################################################################
# fasta2kmerized dict
#
# Take a DNA string, turn it into a positional kmer representation.  This is a 
# 4^k by seq_len / k matrix that is one-hot encoded for tokens.  
#
# I've found this can be quite slow for some reason, even for a single fasta
# entry.
################################################################################
def dna_one_hot_kmer(seq, kmer_length, seq_len=None, flatten=True, keep_repeats=True):
    
    if keep_repeats:
        seq = seq.upper()
    
    #ktable = khmer.new_ktable(kmer_length)
    #kmers_position = {ktable.reverse_hash(i): i for i in range(0, ktable.n_entries())}
        
    bases = ['A','C','G','T']
    kmers = [''.join(p) for p in itertools.product(bases, repeat=kmer_length)]
    kmers_position = {kmers[i]: i for i in range(0, len(kmers))}  
    
    my_kmers = [seq[i:i+kmer_length] for i in range(0, len(seq) - kmer_length + 1, 1)]  ## TODO: might have to revisit this to handle padded sequences
    seq_code = np.zeros((int(pow(4, kmer_length)), len(seq) - kmer_length + 1), dtype='uint8')    
    for i, kmer in enumerate(my_kmers):
        try:
            seq_code[kmers_position[kmer],i] = 1
        except:
            print('Failed to encode the following kmer: ', kmer, file=sys.stderr)
        
    if flatten:
        seq_vec = seq_code.flatten()[None,:]
    
    return seq_vec


################################################################################
# hash_scores
#
# Input
#  scores_file:
#
# Output
#  seq_scores:  Dict mapping FASTA headers to score vectors.
################################################################################
def hash_scores(scores_list):
    seq_scores = {}

    for line in scores_list:
        a = line.split()

        try:
            seq_scores[a[0]] = np.array([float(a[i]) for i in range(1,len(a))])
        except:
            print('Ignoring header line', file=sys.stderr) 

    # consider converting the scores to integers
    int_scores = True
    for header in seq_scores:
        if not np.equal(np.mod(seq_scores[header], 1), 0).all():
            int_scores = False
            break

    if int_scores:
        for header in seq_scores:
            seq_scores[header] = seq_scores[header].astype('uint8')

    return seq_scores


################################################################################
# hash_sequences_1hot  
# Input
#  seq_list:  List of sequences.
#  header_list: List of headers from FASTA file
#  extend_len:  Extend the sequences to this length.
#  dump_N: keep or discard uncertain nucleotides.
#
# Output
#  seq_vecs:    Dict mapping FASTA headers to sequence representation vectors.
################################################################################
def hash_sequences_1hot(seq_list, header_list, extend_len=None, kmer_length=1, dump_N=True):
    # do we need to kmerize?
    kmerize = kmer_length > 1
   
    # determine longest sequence
    if extend_len is not None:
        seq_len = extend_len
    else:
        seq_len = max([len(seq) for seq in seq_list])

    # load and code sequences
    seq_vecs = OrderedDict()
    for seq, header in zip(seq_list, header_list): 
        if 'N' in seq.upper() and dump_N:
            continue
        if seq and not kmerize:
            seq_vecs[header] = dna_one_hot(seq, seq_len)
        if seq and kmerize:
            seq_vecs[header] = dna_one_hot_kmer(seq, kmer_length)        
            
    return seq_vecs


def read_fasta_chunk(fasta_handle, size):
    ''' Read size header,seq records from fasta_handle, return in separate lists.  '''
    mixed_list = list(itertools.islice(fasta_handle, int(size * 2)))
    header_list = [mixed_list[i].strip() for i in range(0,len(mixed_list),2)]
    header_list = [elem[1:] for elem in header_list]    # drop the '>' character which does not appear in the targets
    seq_list = [mixed_list[i].strip() for i in range(1,len(mixed_list),2)]
    return seq_list, header_list

    

################################################################################
# load_data_1hot    
# Input
#  fasta_file:  file handle to FASTA file.
#  scores_file: file handle to scores file.
#  chunksize: size of record to read
#
# Output
#  train_seqs:    Matrix with sequence vector rows.
#  train_scores:  Matrix with score vector rows.
################################################################################
def load_data_1hot(fasta_file, scores_file, chunksize, extend_len=None, mean_norm=True, whiten=False, permute=True, sort=False, kmerize=1):
    
    seq_list, header_list = read_fasta_chunk(fasta_file, chunksize)
    scores_list = list(itertools.islice(scores_file, int(chunksize)))
    
    # load sequences
    seq_vecs = hash_sequences_1hot(seq_list, header_list, extend_len, kmerize)

    # load scores
    seq_scores = hash_scores(scores_list)

    # align and construct input matrix
    train_seqs, train_scores = align_seqs_scores_1hot(seq_vecs, seq_scores, sort)

    # whiten scores
    if whiten:
        train_scores = preprocessing.scale(train_scores)
    elif mean_norm:
        train_scores -= np.mean(train_scores, axis=0)

    # randomly permute
    if permute:
        order = npr.permutation(train_seqs.shape[0])
        train_seqs = train_seqs[order]
        train_scores = train_scores[order]

    return train_seqs, train_scores, header_list


################################################################################
# load_sequences
#
# Input
#  fasta_file:  Input FASTA file.
#
# Output
#  train_seqs:    Matrix with sequence vector rows.
#  train_scores:  Matrix with score vector rows.
################################################################################
def load_sequences(fasta_file, permute=False):
    # load sequences
    seq_vecs = hash_sequences_1hot(fasta_file)

    # stack
    train_seqs = np.vstack(seq_vecs.values())

    # randomly permute the data
    if permute:
        order = npr.permutation(train_seqs.shape[0])
        train_seqs = train_seqs[order]

    return train_seqs



def vecs2dna(seq_vecs):
    ''' vecs2dna

    Input:
        seq_vecs:
    Output:
        seqs
    '''

    # possibly reshape
    if len(seq_vecs.shape) == 2:
        seq_vecs = np.reshape(seq_vecs, (seq_vecs.shape[0], 4, -1))
    elif len(seq_vecs.shape) == 4:
        seq_vecs = np.reshape(seq_vecs, (seq_vecs.shape[0], 4, -1))

    seqs = []
    for i in range(seq_vecs.shape[0]):
        seq_list = ['']*seq_vecs.shape[2]
        for j in range(seq_vecs.shape[2]):
            if seq_vecs[i,0,j] == 1:
                seq_list[j] = 'A'
            elif seq_vecs[i,1,j] == 1:
                seq_list[j] = 'C'
            elif seq_vecs[i,2,j] == 1:
                seq_list[j] = 'G'
            elif seq_vecs[i,3,j] == 1:
                seq_list[j] = 'T'
            elif seq_vecs[i,:,j].sum() == 1:
                seq_list[j] = 'N'
            else:
                print('Malformed position vector: ', seq_vecs[i,:,j], 'for sequence %d position %d' % (i,j), file=sys.stderr)
        seqs.append(''.join(seq_list))
    return seqs


def one_hot_to_kmerized(seq_vecs, kmer_size):
    ''' Take one-hot encoded sequences vectors, return the positional kmerized encoded versions of the same'''
    sequences = decode_one_hot(seq_vecs)
    one_hot_kmer_list = [dna_one_hot_kmer(seq, kmer_size) for seq in sequences]
    return np.asarray(one_hot_kmer_list) 
    

def decode_one_hot(seq_vecs):
    '''
    Input: 
        one hot encoded vectors of shape (num_cases, 4, 1, len_seqs)
    Output:
        character string representation of decoded one-hot vectors
    '''
    return kmer_vecs_to_dna(seq_vecs, k=1)

def kmer_vecs_to_dna(seq_vecs, k):
    '''
    Input: 
        seq_vecs, an nd-array either shaped as either:
            (1) a flattened vector representation (#seqs, 4^k * |sequence| / k)
            (2) a torch tensor representation (#seqs, 4^k, 1, |sequence| / k)
    
    Output: 
        a list of decoded (4^k, |sequence| / k) DNA sequences
    
    '''
    #ktable = khmer.new_ktable(k)
    #kmer_decoder = {i: ktable.reverse_hash(i) for i in range(0, ktable.n_entries())}
    
    bases = ['A','C','G','T']
    kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]   
    kmer_decoder = {i: kmers[i] for i in range(0, len(kmers))}
    
    alphabet_size = int(pow(4, k))
    
    # reshape into a 3-tensor of (sequence, alphabet, position)
    if np.ndim(seq_vecs) == 2:
        n_seqs, flat_seq = seq_vecs.shape
        kmer_cols = flat_seq / alphabet_size
        seq_mats = np.reshape(seq_vecs, (n_seqs, alphabet_size, kmer_cols))
        
    elif np.ndim(seq_vecs) == 4: 
        n_seqs, alphabed_size, useless, kmer_cols = seq_vecs.shape   # seqs.reshape((seqs.shape[0], alphabet_size,1,seqs.shape[1] / alphabet_size)
        seq_mats = np.reshape(seq_vecs, (n_seqs, alphabet_size, kmer_cols))
    
    seqs = []
    
    # Numpy knows to iterate over the first dimension of the (seqs,alphabet,kmer_cols) nd-array.
    for seq_mat in seq_mats:
        seqs.append(kmer_to_dna(seq_mat, kmer_decoder, alphabet_size))
    
    return seqs


def kmer_to_dna(seq, decoder, alphabet_size):
    ''' 
    Inputs: 
        seq: the |alphabet| x |kmerized sequence length| one-hot encoded sequence
        decoder: dict of {position: k-mer string}
        alphabet_size: number of k-mers
        
    Outputs: 
        decoded character string
    '''
    # get the indices of non-zero rows.  np.nonzero works differently than I tought, have to iterate on columns of seq independently
    decoded_kmers = [decoder[np.nonzero(r)[0][0]] for r in np.transpose(seq)]
    firsts = [e[0] for e in decoded_kmers]  # since we're tiling the kmers now, just take the first letter of each decoded kmer
    firsts.extend(decoded_kmers[-1][1:])    # but append the last two characters of the last kmer
    return ''.join(firsts)