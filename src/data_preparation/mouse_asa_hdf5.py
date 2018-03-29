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
from pyfaidx import Fasta
from process_flanks import make_flanks

################################################################################
# mouse_asa_hdf5.py
#
# Make an HDF5 file for tensor input out of a FASTA file and targets text file,
# dividing the data into training, validation, and test data sets
#
################################################################################

################################################################################
# main
################################################################################
def encode_sequences(my_args=None):
    usage = 'usage: %prog [options] <maternal_fasta_file> <paternal_fasta_file> <targets_dir> <out_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Align sizes with batch size')
    parser.add_option('-x', dest='exclude_peaks', default=None, type='string', help='Path to peaks that should be excluded')
    parser.add_option('-e', dest='extend_length', type='int', default=None, help='Extend all sequences to this length [Default: %default]')
    parser.add_option('-r', dest='permute', default=False, action='store_true', help='Permute sequences [Default: %default]')
    parser.add_option('-s', dest='random_seed', default=1, type='int', help='numpy.random seed [Default: %default]')
    parser.add_option('-t', dest='test_chrom', default=6, type='string', help='Test % [Default: %default]')
    parser.add_option('-k', dest='kmerize', default=1, type='int', help='produce kmer-ized representation of the input for this value of k')
    parser.add_option('-v', dest='valid_chrom', default=19, type='string', help='Validation % [Default: %default]')
    parser.add_option('-l', dest='columns', default=300, type='int', help='number of bases (i.e feature columns) in the input')
    parser.add_option('-u', dest='chunks', default=10, type='int', help='Process the fasta file in this many chunks to conserve RAM')
    parser.add_option('-f', dest='flanks', default=True, help='Generate flanking sequences?')
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
    maternal_fasta_file = os.path.expanduser(maternal_fasta_file)
    paternal_fasta_file = os.path.expanduser(paternal_fasta_file)
    b6_atlas_handle = open(maternal_fasta_file, 'r')
    cast_atlas_handle = open(paternal_fasta_file, 'r')
    b6_fasta = Fasta(os.path.join(os.path.dirname(maternal_fasta_file),"Mus_musculus.GRCm38.dna.primary_assembly.fa"))
    cast_fasta = Fasta(os.path.join(os.path.dirname(paternal_fasta_file),"cast.fa"))
 
    #################################################################
    # construct hdf5 representation
    #################################################################
    h5f = h5py.File(out_file, 'w')
    
    # get pandas df for targets
    target_df = process_target_celltype_no_totals(os.path.expanduser(targets_dir))
    group = h5f.require_group(options.group)
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
    
    # exclude any peaks if provided
    if options.exclude_peaks:
        exclude_list = [l.strip() for l in open(os.path.join(os.path.dirname(maternal_fasta_file),options.exclude_peaks),'r').readlines()]
        exclude_re = '|'.join([e.lstrip('>') for e in exclude_list])
        validation_targets = validation_targets[validation_targets["peakID"].str.match(exclude_re) == False]
        test_targets = test_targets[test_targets["peakID"].str.match(exclude_re) == False]
        train_targets = train_targets[train_targets["peakID"].str.match(exclude_re) == False]
    else:
        exclude_list = []

    #################################################################
    # load data for each of train, test, validation
    #################################################################
    
    b6_headers = []
    b6_sequences = []
    cast_headers = []
    cast_sequences = []
    
    for b6_fasta_record, cast_fasta_record in zip(_generate_fasta(b6_atlas_handle), _generate_fasta(
        cast_atlas_handle)):
        
        if len(b6_fasta_record) != 2 or len(cast_fasta_record) != 2:
            continue
    
        header, seq = b6_fasta_record
            
        if header in exclude_list:
            continue
            
        b6_headers.append(header.lstrip(">"))
        b6_sequences.append(seq)
            
        header, seq = cast_fasta_record
        cast_headers.append(header.lstrip(">"))
        cast_sequences.append(seq)
        
    
    b6_data = pd.DataFrame.from_dict({'header': b6_headers, 'sequences': b6_sequences})
    b6_test = b6_data[b6_data["header"].str.startswith(str(options.test_chrom))]
    b6_valid = b6_data[b6_data["header"].str.startswith(str(options.valid_chrom))]
    b6_train = b6_data[(b6_data["header"].str.startswith(str(options.valid_chrom)) == False) & (b6_data["header"].str.startswith(str(options.test_chrom)) == False)]
    
    cast_data = pd.DataFrame.from_dict({'header': cast_headers, 'sequences': cast_sequences})
    cast_test = cast_data[cast_data["header"].str.startswith(str(options.test_chrom))]
    cast_valid = cast_data[cast_data["header"].str.startswith(str(options.valid_chrom))]
    cast_train = cast_data[(cast_data["header"].str.startswith(str(options.valid_chrom)) == False) & (cast_data["header"].str.startswith(str(options.test_chrom)) == False)]    
    
    # make flank sequences
    if options.flanks:
        
        # make up peak bed file
        with open('temp_peak_bedfile.bed','w') as f:
            bedlines = convert_headers_to_bedfile(b6_headers)
            for line in bedlines:
                print(line, file=f)
                
        # assemble args, determine flank regions
        arg_string = "-o temp_flank_bedfile.bed " + "-s " + str(options.columns) + ' -l mouse.mm10.genome ' +'temp_peak_bedfile.bed'
        my_args = arg_string.split(sep=' ')
        make_flanks(my_args)            

        # harvest and encode seqs
        with open('temp_flank_bedfile.bed','r') as f:
            flanks = f.readlines()
        
        b6_training_flank_seqs = []
        b6_testing_flank_seqs = []
        b6_validation_flank_seqs = []
        cast_training_flank_seqs = []
        cast_testing_flank_seqs = []
        cast_validation_flank_seqs = []
        
        for flank in flanks:
            chrom, start, end, *_ = flank.split()
            chrom = chrom.lstrip("chr")
            if chrom == options.test_chrom:
                b6_testing_flank_seqs.append(b6_fasta[chrom][int(start):int(end)].seq)
                cast_testing_flank_seqs.append(cast_fasta[chrom][int(start):int(end)].seq)                
            elif chrom == options.valid_chrom:
                b6_validation_flank_seqs.append(b6_fasta[chrom][int(start):int(end)].seq)
                cast_validation_flank_seqs.append(cast_fasta[chrom][int(start):int(end)].seq)                
            else:
                b6_training_flank_seqs.append(b6_fasta[chrom][int(start):int(end)].seq)
                cast_training_flank_seqs.append(cast_fasta[chrom][int(start):int(end)].seq)
        
        b6_train_flanks = onehot_encode_sequences(b6_training_flank_seqs)
        b6_test_flanks = onehot_encode_sequences(b6_testing_flank_seqs)
        b6_valid_flanks = onehot_encode_sequences(b6_validation_flank_seqs)
        
        cast_train_flanks = onehot_encode_sequences(cast_training_flank_seqs)
        cast_test_flanks = onehot_encode_sequences(cast_testing_flank_seqs)
        cast_valid_flanks = onehot_encode_sequences(cast_validation_flank_seqs)
        
        # tidy up peak, flank bedfiles
        os.remove("temp_peak_bedfile.bed")
        os.remove("temp_flank_bedfile.bed")
            
    
    # encode training seqs
    b6_train_sequences = onehot_encode_sequences([resize_sequences(coords, sequence, b6_fasta) for coords,sequence in zip(
        b6_train["header"], 
        b6_train["sequences"])])
    cast_train_sequences = onehot_encode_sequences([resize_sequences(coords, sequence, cast_fasta) for coords,sequence in zip(
        cast_train["header"], 
        cast_train["sequences"])])  
    
    # encode test seqs
    b6_test_sequences = onehot_encode_sequences([resize_sequences(coords, sequence, b6_fasta) for coords,sequence in zip(
        b6_test["header"], 
        b6_test["sequences"])]) 
    cast_test_sequences = onehot_encode_sequences([resize_sequences(coords, sequence, cast_fasta) for coords,sequence in zip(
        cast_test["header"], 
        cast_test["sequences"])])
    
    # encode valid seqs
    b6_valid_sequences = onehot_encode_sequences([resize_sequences(coords, sequence, b6_fasta) for coords,sequence in zip(
        b6_valid["header"], 
        b6_valid["sequences"])])
    cast_valid_sequences = onehot_encode_sequences([resize_sequences(coords, sequence, cast_fasta) for coords,sequence in zip(
        cast_valid["header"], 
        cast_valid["sequences"])])  

    # reshape sequences, combine with flanks if needed, write to h5 file
    
    b6_train_sequences = b6_train_sequences.reshape((b6_train_sequences.shape[0],4,1,b6_train_sequences.shape[-1]))
    cast_train_sequences = cast_train_sequences.reshape((cast_train_sequences.shape[0],4,1,cast_train_sequences.shape[-1]))
    b6_test_sequences = b6_test_sequences.reshape((b6_test_sequences.shape[0],4,1,b6_test_sequences.shape[-1]))
    cast_test_sequences = cast_test_sequences.reshape((cast_test_sequences.shape[0],4,1,cast_test_sequences.shape[-1]))    
    b6_valid_sequences = b6_valid_sequences.reshape((b6_valid_sequences.shape[0],4,1,b6_valid_sequences.shape[-1]))
    cast_valid_sequences = cast_valid_sequences.reshape((cast_valid_sequences.shape[0],4,1,cast_valid_sequences.shape[-1]))    
    
    if options.flanks:
        b6_train_flanks = b6_train_flanks.reshape((b6_train_flanks.shape[0],4,1,b6_train_flanks.shape[-1]))
        cast_train_flanks = cast_train_flanks.reshape((cast_train_flanks.shape[0],4,1,cast_train_flanks.shape[-1]))
        b6_test_flanks = b6_test_flanks.reshape((b6_test_flanks.shape[0], 4, 1, b6_test_flanks.shape[-1]))
        cast_test_flanks = cast_test_flanks.reshape((cast_test_flanks.shape[0], 4, 1, cast_test_flanks.shape[-1]))
        b6_valid_flanks = b6_valid_flanks.reshape((b6_valid_flanks.shape[0],4,1,b6_valid_flanks.shape[-1]))
        cast_valid_flanks = cast_valid_flanks.reshape((cast_valid_flanks.shape[0],4,1,cast_valid_flanks.shape[-1]))
        
        both_allele_train_seqs = np.vstack((b6_train_sequences, cast_train_sequences, b6_train_flanks, cast_train_flanks))
        training_targets = np.vstack((target_df_to_array(train_targets),np.zeros((b6_train_flanks.shape[0],5)),np.zeros((cast_train_flanks.shape[0],5))))
        both_allele_test_seqs = np.vstack((b6_test_sequences, cast_test_sequences, b6_test_flanks, cast_test_flanks))
        testing_targets = np.vstack((target_df_to_array(test_targets),np.zeros((b6_test_flanks.shape[0],5)),np.zeros((cast_test_flanks.shape[0],5))))
        both_allele_valid_seqs = np.vstack((b6_valid_sequences, cast_valid_sequences, b6_valid_flanks, cast_valid_flanks))
        validation_targets = np.vstack((target_df_to_array(validation_targets),np.zeros((b6_valid_flanks.shape[0],5)),np.zeros((cast_valid_flanks.shape[0],5))))
    
    else:
        both_allele_train_seqs = np.vstack((b6_train_sequences, cast_train_sequences))
        training_targets = target_df_to_array(train_targets)
        both_allele_test_seqs = np.vstack((b6_test_sequences, cast_test_sequences))
        testing_targets = target_df_to_array(test_targets)     
        both_allele_valid_seqs = np.vstack((b6_valid_sequences, cast_valid_sequences))
        validation_targets = target_df_to_array(validation_targets)        
    
    if options.permute:
        order = np.random.permutation(both_allele_train_seqs.shape[0])
        both_allele_train_seqs = both_allele_train_seqs[order]
        training_targets = training_targets[order]
        
        order = np.random.permutation(both_allele_test_seqs.shape[0])
        both_allele_test_seqs = both_allele_test_seqs[order]
        testing_targets = testing_targets[order]
        
        order = np.random.permutation(both_allele_valid_seqs.shape[0])
        both_allel_valid_seqs = both_allele_valid_seqs[order]
        validation_targets = validation_targets[order]  
    
    valid_count = both_allele_valid_seqs.shape[0]
    test_count = both_allele_test_seqs.shape[0]
    train_count = both_allele_train_seqs.shape[0]
    
    if train_count > 0:
        train_in_dset = data_group.create_dataset('train_in', shape=(train_count, alphabet_size, 1, feature_cols), dtype='float')
        train_out_dset = label_group.create_dataset('train_out', shape=training_targets.shape, dtype='uint8')

    if valid_count > 0:
        valid_in_dset = data_group.create_dataset('valid_in', shape=(valid_count, alphabet_size, 1, feature_cols), dtype='float')
        valid_out_dset = label_group.create_dataset('valid_out', shape=validation_targets.shape, dtype='uint8')

    if test_count > 0: 
        test_in_dset = data_group.create_dataset('test_in', shape=(test_count, alphabet_size, 1, feature_cols),dtype='float')
        test_out_dset = label_group.create_dataset('test_out', shape=testing_targets.shape, dtype='uint8')    
    
    train_in_dset[:] = both_allele_train_seqs
    train_out_dset[:] = training_targets
    test_in_dset[:] = both_allele_test_seqs
    test_out_dset[:] = testing_targets
    valid_in_dset[:] = both_allele_valid_seqs
    valid_out_dset[:] = validation_targets
    
    # tidy up open files
    b6_atlas_handle.close()
    cast_atlas_handle.close()
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


def convert_headers_to_bedfile(lines):
    ''' convert coordinates in IGV syntax (chr:start-end) into bed coordinates '''
    bed_lines = []
    for line in lines:
        chrom, s_e = line.split(":")
        start, end = s_e.split("-")
        bed_lines.append("\t".join(["chr"+chrom, start, end]))
    return bed_lines
    

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



def target_df_to_array(df):
    ''' Split a dataframe with B6 and CAST counts into two separate arrays, 
    then return as one stacked array '''
    b6_counts = df[df.columns[df.columns.to_series().str.contains('b6')]]
    cast_counts = df[df.columns[df.columns.to_series().str.contains('cast')]]
    return np.vstack((b6_counts.as_matrix(), cast_counts.as_matrix()))


def onehot_encode_sequences(sequences):
    ''' Encode each sequence in sequences, without any expansion or truncation '''
    onehot = []
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
    for sequence in sequences:
        arr = np.zeros((len(sequence), 4)).astype("float")
        for (i, letter) in enumerate(sequence):
            if letter is not 'N':
                arr[i, mapping[letter]] = 1.0
            else:
                arr[i,:] = 0.25
        onehot.append(arr.transpose())
        
    return np.stack(onehot)


def flank_to_sequence(flank, fasta_handle):
    ''' Extract the specified flanking sequence from the provided fasta 
    Presumes that the flanking sequence is encoded in BED format '''
    chrom, start, end = flank.split()
    chrom = chrom.lstrip("chr")
    fasta_seq = fasta[chrom][int(start):int(end)]
    return fasta_seq.seq
   
def resize_sequences(coords, sequence, fasta, size=300):
    ''' Resize the sequence at coords, truncating or expanding (from fasta)
    file as required '''
    
    if len(sequence) < size:
        
        # calculate border of flaking sequence
        coords = coords.split('|')[0]
        chrom, s_e = coords.lstrip('>').split(':')
        start, end = s_e.split('-')
        
        flank_size = size - (int(end) - int(start))
        five_prime_flank_size = np.random.binomial(flank_size, 0.5)
        three_prime_flank_size = flank_size - five_prime_flank_size
        start = int(start) - five_prime_flank_size
        end = int(end) + three_prime_flank_size
        
        # extract the allele specific sequence from the fasta file
        fasta_seq = fasta[chrom][int(start):int(end)]
        resized_sequence = fasta_seq.seq
        
    else:
        # select random subset, unless we're right on 300bp
        start = np.random.randint(0,len(sequence) - size) if len(sequence) - size > 0 else 0
        end = start + size
        resized_sequence = sequence[start:end]
    
    assert( len(resized_sequence) == 300)
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
    arg_string = "-t 6 -v 19 -b 20 -x N_containing_peaks.txt -r ~/projects/SeqDemote/data/ATAC/mouse_asa/sequences/CD8_effector/atlas.Ref.fa ~/projects/SeqDemote/data/ATAC/mouse_asa/sequences/CD8_effector/atlas.Alt.fa ~/projects/SeqDemote/data/ATAC/mouse_asa/mapped_reads/CD8_effector/ mouse_asa.h5"
    encode_sequences(arg_string.split())
