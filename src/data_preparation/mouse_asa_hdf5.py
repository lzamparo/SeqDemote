#!/usr/bin/env python
from __future__ import print_function
from optparse import OptionParser
import sys
import re
import os

import pandas as pd
import h5py
import numpy.random as npr
import numpy as np

import subprocess

from utils import dna_io

################################################################################
# mouse_asa_hdf5.py
#
# Make an HDF5 file for tensor input out of a FASTA file and targets text file,
# dividing the data into training, validation, and test data sets
#
# Usage is like 
# seq_hdf5.py -c -r -t 71886 -v 70000 encode_roadmap.fa encode_roadmap_act.txt encode_roadmap.h5
################################################################################

################################################################################
# main
################################################################################
def encode_sequences(my_args=None):
    usage = 'usage: %prog [options] <maternal_fasta_file> <paternal_fasta_file> <targets_dir> <out_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Align sizes with batch size')
    parser.add_option('-e', dest='extend_length', type='int', default=None, help='Extend all sequences to this length [Default: %default]')
    parser.add_option('-r', dest='permute', default=False, action='store_true', help='Permute sequences [Default: %default]')
    parser.add_option('-s', dest='random_seed', default=1, type='int', help='numpy.random seed [Default: %default]')
    parser.add_option('-t', dest='test_chrom', default=6, type='string', help='Test % [Default: %default]')
    parser.add_option('-k', dest='kmerize', default=1, type='int', help='produce kmer-ized representation of the input for this value of k')
    parser.add_option('-v', dest='valid_chrom', default=21, type='string', help='Validation % [Default: %default]')
    parser.add_option('-l', dest='columns', default=600, type='int', help='number of bases (i.e feature columns) in the input')
    parser.add_option('-u', dest='chunks', default=10, type='int', help='Process the fasta file in this many chunks to conserve RAM')
    parser.add_option('-g', dest='group', default='/', type='str', help='All data (both encoded sequences and activation labels) are stored underneath this group.  Will be created if it does not arleady exist.')
    if not my_args:
        (options,args) = parser.parse_args()
    else:
        (options,args) = parser.parse_args(args=my_args)
    if len(args) != 4:
        parser.error('Must provide both maternal and paternal fasta files, targets directory, and an output prefix')
    else:
        maternal_fasta_file = args[0]
        paternal_fasta_file = args[1]
        targets_dir = args[2]
        out_file = args[3]

    # seed rng before shuffle
    npr.seed(options.random_seed)

    #################################################################
    # prepare bookeeping data for processing the sizes of each chunk
    # of the fasta input
    #################################################################    
    b6_fasta_handle = open(os.path.expanduser(maternal_fasta_file), 'r')
    cast_fasta_handle = open(os.path.expanduser(paternal_fasta_file), 'r')
 
    
    #################################################################
    # construct hdf5 representation
    #################################################################
    h5f = h5py.File(out_file, 'a')
    
    # get pandas df for targets
    target_df = process_target_celltype_no_totals(os.path.expanduser(targets_dir))
    target_labels = np.array(list(target_df)).astype('|S21')

    group = h5f.create_group(options.group)
    group.create_dataset('target_labels', data=target_labels)
    data_group = group.create_group('data')
    label_group = group.create_group('labels')    

    alphabet_size = 4
    feature_cols = options.columns
    if options.kmerize > 1:
        alphabet_size = int(pow(4, options.kmerize))
        feature_cols = options.columns - options.kmerize + 1
        
    # need size and shape estimates of test, valid sets here
    validation_targets = target_df[target_df["peakID"].str.startswith(str(options.valid_chrom))]
    test_targets = target_df[target_df["peakID"].str.startswith(str(options.test_chrom))]
    train_targets = target_df[(target_df["peakID"].str.startswith(str(options.test_chrom)) == False) & (target_df["peakID"].str.startswith(str(options.valid_chrom)) == False)]
    
    valid_count = validation.targets.shape[0] * 2       # double is due to one data point for b6, cast each
    test_count = test_targets.shape[0] * 2
    train_count = (target_df.shape[0] - valid_count - test_count) * 2
    
    if train_count > 0:
        train_in_dset = data_group.create_dataset('train_in', shape=(train_count, alphabet_size, 1, feature_cols), dtype='uint8')
        train_out_dset = label_group.create_dataset('train_out', shape=(train_count, len(target_labels)), dtype='uint8')

    if valid_count > 0:
        valid_in_dset = data_group.create_dataset('valid_in', shape=(valid_count, alphabet_size, 1, feature_cols), dtype='uint8')
        valid_out_dset = label_group.create_dataset('valid_out', shape=(valid_count, len(target_labels)), dtype='uint8')

    if test_count > 0: 
        test_in_dset = data_group.create_dataset('test_in', shape=(test_count, alphabet_size, 1, feature_cols),dtype='uint8')
        test_out_dset = label_group.create_dataset('test_out', shape=(test_count, len(target_labels)), dtype='uint8')

    #################################################################
    # load data for each of train, test, validation
    #################################################################
    
    b6_headers = []
    b6_sequences = []
    cast_headers = []
    cast_sequences = []
    
    for b6_fasta_record, cast_fasta_record in zip(_generate_fasta(b6_fasta_handle), _generate_fasta(
        cast_fasta_handle)):
        
        if len(b6_fasta_record) == 2:
            header, seq = b6_fasta_record
            b6_headers.append(header.lstrip(">"))
            b6_sequences.append(seq)
            
        if len(cast_fasta_record == 2):
            header, seq = cast_fasta_record
            cast_headers.append(header.lstrip(">"))
            cast_sequences.append(seq)
            
    
    b6_data = pd.DataFrame.from_dict({'header': b6_headers, 'sequences': b6_sequences})
    b6_test = b6_data[b6_data["header"].str.startswith(str(options.test_chrom))]
    b6_valid = b6_data[b6_data["header"].str.startswith(str(options.valid_chrom))]
    b6_train = b6_data[(b6_data["header"].str.startswith(str(options.valid_chrom)) == False) & (b6_data["header"].str.startswith(str(options.test_chrom)) == False)]
    
    cast_data = pd.DataFrame.from_dit({'header': cast_headers, 'sequences': cast_sequences})
    cast_test = cast_data[cast_data["header"].str.startswith(str(options.test_chrom))]
    cast_valid = cast_data[cast_data["header"].str.startswith(str(options.valid_chrom))]
    cast_train = cast_data[(cast_data["header"].str.startswith(str(options.valid_chrom)) == False) & (cast_data["header"].str.startswith(str(options.test_chrom)) == False)]    
    
    # encode training seqs
    
    
    # encode test seqs
    
    
    # encode valid seqs




    #seqs, targets, headers = dna_io.load_data_1hot(fasta_handle, targets_handle, chunksize, extend_len=options.extend_length, mean_norm=False, whiten=False, permute=False, sort=False, kmerize=options.kmerize)


    # reshape sequences appropriately, depending on k.  Assume that k is specified to divide 600.
    seqs = seqs.reshape((seqs.shape[0],4,1,seqs.shape[1]/4))

    if options.permute:
        order = npr.permutation(seqs.shape[0])
        seqs = seqs[order]
        targets = targets[order]
        #headers = headers[order]

    # tidy up open files
    fasta_handle.close()
    targets_handle.close()
    h5f.close()
    


def _generate_fasta(file):
    ''' Parse and yield two line fasta records '''
    record = []
    for line in file:
        if line.startswith(">"):
            if record:
                yield record
            record = [line.strip()]
        else:
            record.append(line.strip())
    yield record

    
def parse_peak(l):
    coords,counts = l.split()
    maternal, paternal = counts.split(';')
    return coords, maternal, paternal


def scrape_name(filename, get_rep_num):
    m = get_rep_num.match(filename)
    return m.groups()[0]


def common_reads(total_read_lines, asa_snp_reads):
    ''' Calculates the read coverage that can plausibly be shared between maternal and paternal chromosomes. 
    We assume here that for a given peak, the total read coverage for that peak that cannot be assigned to
    either maternal or paternal '''
    
    common_reads = []
    for l,a in zip(total_read_lines, asa_snp_reads):
        coords, count = l.split()
        reads = int(count) - a
        common_reads.append(reads)
        
    return common_reads


def process_target_celltype_no_totals(ct_dir):
    ''' Take a list of cell type replicates within a cell type directory, 
    produce a target table of counts.  Ignore the total counts, just 
    use the differentially accessible reads '''
    ct_name = os.path.basename(ct_dir.strip('/'))
    differential_reps = [f for f in os.listdir(ct_dir) if f.endswith("peak.cnt")]
    get_rep_num = re.compile('.*\_r([\d])\_.*')

    dfs = []
    for r in differential_reps:
        repname = ct_name + "_rep" + scrape_name(r,get_rep_num) + "_"
        with open(os.path.join(ct_dir,r), 'r') as d_file:
            d_lines = [l.strip() for l in d_file.readlines()]
            coords = [c for (c,m,p) in [parse_peak(l) for l in d_lines]]
            maternal = [int(m) for (c,m,p) in [parse_peak(l) for l in d_lines]]
            paternal = [int(p) for (c,m,p) in [parse_peak(l) for l in d_lines]]
            
        df = pd.DataFrame({"peakID": coords, repname+"b6": maternal, repname+"cast": paternal})
        
        dfs.append(df)
    
    # merge all dfs on peakID
    peak_stats = dfs[0]
    for df in dfs[1:]:
        peak_stats = pd.merge(peak_stats, df, on="peakID")
    
    # reorder columns
    cols = list(peak_stats)
    ordered_cols = [cols[2]]
    ordered_cols.extend(cols[0:2])
    ordered_cols.extend(cols[3:])
    return peak_stats.ix[:, ordered_cols]    

def process_target_celltype(ct_dir):
    ''' Take a list of cell type replicates within a cell type directory, 
    produce a target table of counts 
    
    peakID	H1hesc	CD34	CD14	CD56	CD3	CD19
chr1:1208992-1209592(+)	1	1	1	1	1	1
chr1:11120062-11120662(+)	1	1	1	1	1	1
chr1:161067622-161068222(+)	1	1	1	1	1	1
chr4:84376575-84377175(+)	1	1	1	1	1	1
    '''
    
    ct_name = os.path.basename(ct_dir.strip('/'))
    differential_reps = [f for f in os.listdir(ct_dir) if f.endswith("peak.cnt")]
    total_count_reps = [f for f in os.listdir(ct_dir) if f.endswith("total.cnt")]
    get_rep_num = re.compile('.*\_r([\d])\_.*')

    dfs = []
    for r,t in zip(differential_reps, total_count_reps):
        repname = ct_name + "_rep" + scrape_name(r,get_rep_num) + "_"
        with open(os.path.join(ct_dir,r), 'r') as d_file, open(os.path.join(ct_dir,t), 'r') as t_file:
            d_lines = [l.strip() for l in d_file.readlines()]
            t_lines = [l.strip() for l in t_file.readlines()]
            coords = [c for (c,m,p) in [parse_peak(l) for l in d_lines]]
            maternal = [int(m) for (c,m,p) in [parse_peak(l) for l in d_lines]]
            paternal = [int(p) for (c,m,p) in [parse_peak(l) for l in d_lines]]
            
            asa_snp_sum = [m + p for (m,p) in zip(maternal, paternal)]
            common = common_reads(t_lines, asa_snp_sum)
            
            maternal = [m + c for m, c in zip(maternal, common)]
            paternal = [p + c for p, c in zip(paternal, common)]
            
        df = pd.DataFrame({"peakID": coords, repname+"b6": maternal, repname+"cast": paternal})
        dfs.append(df)
    
    # merge all dfs on peakID
    peak_stats = dfs[0]
    for df in dfs[1:]:
        peak_stats = pd.merge(peak_stats, df, on="peakID")
    return peak_stats


def onehot_encode_sequences(sequences):
    ''' Encode each sequence in sequences, without any expansion or truncation '''
    onehot = []
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
    for sequence in sequences:
        arr = np.zeros((len(sequence), 4)).astype("float")
        for (i, letter) in enumerate(sequence):
            arr[i, mapping[letter]] = 1.0
        onehot.append(arr)
    return onehot
   
   
def resize_sequences(coords, sequence, fasta, size=300):
    ''' Resize the sequence at coords, truncating or expanding (from fasta)
    file as required '''
    
    if len(sequence) < size:
        pass #TODO: fill in
    
    else:
        pass #TODO: fill in
    
    return resized_sequence
    

def batch_round(count, batch_size):
    if batch_size != None:
        count -= (batch_size % count)
    return count

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    # DEBUG
    arg_string = "-b 20 ~/projects/SeqDemote/data/ATAC/mouse_asa/sequences/CD8_effector/atlas.Ref.fa ~/projects/SeqDemote/data/ATAC/mouse_asa/sequences/CD8_effector/atlas.Alt.fa ~/projects/SeqDemote/data/ATAC/mouse_asa/mapped_reads/CD8_effector/ mouse_asa.h5"
    encode_sequences(arg_string.split())
