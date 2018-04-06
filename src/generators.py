import os
import numpy as np

from utils import dna_io

### Possible data augmentation schemes by subsequence plucking, and kmerization of different varieties

default_augmentation_params = {
    'kmerize': 1,
    'subsequence_range': (-300, 300),
    'subsequence_length': 600,
    'do_subsequences': False,
}

kmer_augmentation_params = {
    'kmerize': 3,
    'subsequence_range': (-300, 300),
    'subsequence_length': 200,
    'do_subsequences': False,
}

subsequence_augmentation_params = {
    'kmerize': 1,
    'subsequence_range': (-500, 500),
    'subsequence_length': 500,
    'do_subsequences': False,
}



def labeled_sequence_gen(sequences, labels, chunk_size=4096, num_chunks=458, rng=np.random):
    ''' Given training data array ref, build and return a generator for one-hot encoded training sequence data '''
    for n in range(num_chunks):
        indices = rng.randint(0, len(sequences), chunk_size)
        
        sequences_rows = sequences.shape[1]
        sequences_cols = sequences.shape[3]
        
        labels_output_shape = labels.shape[1]
        
        chunk_x = np.zeros((chunk_size, sequences_rows, 1, sequences_cols), dtype='float32')
        chunk_y = np.zeros((chunk_size,labels_output_shape), dtype='float32')
        
        for k, idx in enumerate(indices):
            chunk_x[k] = sequences[indices[k]]
            chunk_y[k] = labels[indices[k]]
            
        yield chunk_x, chunk_y
    
def labeled_kmer_sequence_gen(sequences, labels, kmersize=3, chunk_size=4096, num_chunks=458, rng=np.random):
    ''' Given training data array ref, build and return a generator for one-hot encoded kmerized training sequence data '''
    for n in range(num_chunks):
        indices = rng.randint(0, len(sequences), chunk_size)
        
        sequences_rows = sequences.shape[1]
        sequences_cols = sequences.shape[3]
        
        labels_output_shape = labels.shape[1]
        
        chunk_x = np.zeros((chunk_size, pow(sequences_rows,kmersize), 1, sequences_cols - (kmersize-1)), dtype='float32')
        chunk_y = np.zeros((chunk_size, labels_output_shape), dtype='float32')
        
        for k, idx in enumerate(indices):
            data_flat = dna_io.one_hot_to_kmerized(sequences[indices[k]], kmersize)
            chunk_x[k] = data_flat.reshape((data_flat.shape[0], pow(sequences_rows,kmersize), 1, sequences_cols - (kmersize-1)))
            chunk_y[k] = labels[indices[k]]
        
        yield chunk_x, chunk_y
        
        
        

def labeled_kmer_sequence_mismatch_gen(sequences, labels, kmersize=3, chunk_size=4096, num_chunks=458, rng=np.random):
    ''' Given training data array ref, build and return a generator measuring positional mismatches from each kmer '''
    for n in range(num_chunks):
        indices = rng.randint(0, len(sequences), chunk_size)
        
        sequences_rows = sequences.shape[1]
        sequences_cols = sequences.shape[3]
        
        labels_output_shape = labels.shape[1]
        
        chunk_x = np.zeros((chunk_size, pow(sequences_rows, kmersize), 1, sequences_cols - (kmersize - 1)), dtype='float32')
        chunk_y = np.zeros((chunk_size, labels_output_shape), dtype='float32')
        
        for k, idx in enumerate(indices):
            decoded_seqs = dna_io.decode_one_hot(sequences[indices[k]])
            mismatch_seqs = np.asarray([dna_io.dna_mismatch_kmer(seq, kmersize) for seq in decoded_seqs])
            
            chunk_x[k] = mismatch_seqs.reshape((chunk_size, pow(sequences_rows,kmersize), 1, sequences_cols - (kmersize-1)))
            chunk_y[k] = labels[indices[k]]
            
        yield chunk_x, chunk_y
