import os
import argparse
import fnmatch
import pickle
import h5py
import time

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from collections import OrderedDict

def _generate_fasta(file):
    ''' Parse and yield two line fasta records '''
    
    record = []
    for line in file:
        if line.startswith(">"):
            if record:
                yield record[0], ''.join(record[1:])
            record = [line.strip()]
        else:
            record.append(line.strip())
    yield record[0], ''.join(record[1:])
    
def parse_label_line(l):
    ''' Parse a line in the label file to return 
    an array of binary labels '''
    
    l = l.strip()
    parts = l.split()
    return [int(p) for p in parts[1:]]
 
def peak_to_probes(peak, stride):
    ''' given the 300bp peak and stride, return 
    a list of the pseudo probes within the peak '''
    
    return [peak[i*stride : (i*stride)+20] for i in range(0, (len(peak) - 20) // stride)]

def probe_to_visible_kmers(probe, kmer_len=8):
    ''' find all kmers in the given probe, encode them but *not* their wildcard kmer matches 
    In the interests of speed, maybe we can do the conversion to embedded kmers -> probe 
    as a transformation of the dataset
    '''
    kmers = [probe[i:i+kmer_len] for i in range(0, len(probe) - kmer_len + 1, 1)]
    return np.array([kmer_to_id[element] for element in kmers]) 
    

def probe_to_kmers(probe, kmer_len=8):
    ''' find all kmers in the given probe, including the wildcard matching kmers
    return the integer codes for all matches as a list. '''
    
    kmers = [probe[i:i+kmer_len] for i in range(0, len(probe) - kmer_len + 1, 1)]
    matching_bindspace_elements = []
    
    for kmer in kmers:
        ### don't search here, just do the lookup.  Only a handful of kmers in the
        ### wildcard k-mer set can match to each kmer, so just compute and store this
        ### then look up here.
        wildcard_match_set = read_df[read_df['Exact'] == kmer]['Features']
        matching_bindspace_elements.extend(list(wildcard_match_set))
        
    return np.array([kmer_to_id[element] for element in matching_bindspace_elements]) 
    
    
# treat this number of consecutive bases of ATAC-seq peak as a probe to be embedded
probe_len = 20

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", dest='prefix', type=str, help='prefix for where the infile, outfile, pickle should go')
parser.add_argument("--infile", dest='bindspace', type=str, help='bindspace input file')
parser.add_argument("--outfile", dest='h5', type=str, help='h5 output file')
parser.add_argument("--fasta", dest='fasta', type=str, help='input fasta file for ATAC-seq')
parser.add_argument("--labels", dest='labels', type=str, help='label data for ATAC-seq peaks')
parser.add_argument("--picklefile", dest='picklefile', type=str, help='pickle file to store dicts')
parser.add_argument("--pairwisemat", dest='pairwise', type=str, help='pairwise kmer matrix (8-mers -> WC 8-mer map) file')
parser.add_argument("--stride", dest='stride', type=int, default=10, help='sample probes from peaks using the given stride (default: 10)')
args = parser.parse_args()

# parse the code file, generate both kmer -> int, int -> list of kmers, 
# WC kmer -> code dicts
kmer_to_id = OrderedDict()
id_to_wc_kmers = OrderedDict()
wc_kmer_to_vec = OrderedDict()

# parse the sequence 8-mers -> wildcard 8-mer basis file
# can query the df like: read_df[read_df['Exact'] == 'AAAAAAAC']['Features']
# yields Series of matching wildcard 8-mers
with open(os.path.expanduser(os.path.join(args.prefix,args.pairwise)),'rb') as f:
    read_df = feather.read_feather(f)
    
for i, kmer in enumerate(set(read_df['Exact'])):
    kmer_to_id[kmer] = i
    id_to_wc_kmers[i] = list(read_df[read_df['Exact'] == kmer]['Features'])
    

with open(os.path.expanduser(os.path.join(args.prefix, args.bindspace)),'r') as f:
    for line in f:
        line = line.strip()
        parts = line.split('\t')
        kmer, rest = parts[0], parts[1:]
        vec = np.asarray(rest, dtype=np.float)
        wc_kmer_to_vec[kmer] = vec
    


# get fasta records, labels
records = []
with open(os.path.expanduser(os.path.join(args.prefix, args.fasta)), 'r') as f:
    for record in _generate_fasta(f):
        records.append(record)

labels = []
with open(os.path.expanduser(os.path.join(args.prefix, args.labels)), 'r') as f:
    header = f.readline()
    for line in f:
        labels.append(parse_label_line(line))

assert(len(records) == len(labels))
num_peaks = len(records)
label_len = len(labels[0])

# process header for labeling dimensions of label h5 data
dim_labels = [l.rstrip('"').lstrip('"') for l in header.strip().split()]
labels_as_array = np.array(tuple(dim_labels)).astype('|S9')
#
# calculate training / test / validation split
# 0: train, 1: validation, 2: test
train_valid_test_split = np.random.choice([0,1,2], num_peaks, p=[0.8,0.1,0.1])

# for each peak, choose a split, partition into probes, 
# determine the kmer spectrum of the probes, write 
# their ID as an array.
probes_per_peak = (300 - 20 + 1) // args.stride
kmers_per_probe = 20 - 8 + 1

# generate hdf5 file with training, validation, test split peaks
encoded_data = h5py.File(os.path.join(args.prefix, args.h5), 'w',swmr=True)
train_data_group = encoded_data.create_group("/train/data")
valid_data_group = encoded_data.create_group("/valid/data")
test_data_group = encoded_data.create_group("/test/data")
train_label_group = encoded_data.create_group("/train/labels")
valid_label_group = encoded_data.create_group("/valid/labels")
test_label_group = encoded_data.create_group("/test/labels")

# - each dataset (train/test/validation) will be a # peaks x (# probes x # kmers per probe)
# - each probe will be its own dataset
# - each peak is its own group within the train/test/valid groups

num_train_peaks = train_valid_test_split[train_valid_test_split == 0].shape[0]
num_valid_peaks = train_valid_test_split[train_valid_test_split == 1].shape[0]
num_test_peaks = train_valid_test_split[train_valid_test_split == 2].shape[0]

train_data = train_data_group.create_dataset(name='train_data', shape=(num_train_peaks, probes_per_peak, kmers_per_probe))
valid_data = valid_data_group.create_dataset(name='valid_data', shape=(num_valid_peaks, probes_per_peak, kmers_per_probe))
test_data = test_data_group.create_dataset(name='test_data', shape=(num_test_peaks, probes_per_peak, kmers_per_probe))

train_labels = train_label_group.create_dataset(name='train_labels', shape=(num_train_peaks,label_len))
train_labels.attrs['column_names'] = labels_as_array
valid_labels = valid_label_group.create_dataset(name='valid_labels', shape=(num_valid_peaks,label_len))
valid_labels.attrs['column_names'] = labels_as_array
test_labels = test_label_group.create_dataset(name='test_labels', shape=(num_test_peaks,label_len))
test_labels.attrs['column_names'] = labels_as_array

processed_peaks = 0
ttv_data_and_labels = {0: (train_data, train_labels), 1: (valid_data, valid_labels), 2: (test_data, test_labels)}
ttv_processed = {0: 0, 1: 0, 2: 0}

for record, label, split in zip(records, labels, train_valid_test_split):
    # make the sub-group within the appropriate split
    peak_id, sequence = record
    write_index = ttv_processed[split]
    dataset, labelset = ttv_data_and_labels[split]
    
    # write the labels
    labelset[write_index,:] = np.array(label)
    
    # write the encoding of the probes rocess each probe in the peak
    t0 = time.time()
    for probe_index, probe in enumerate(peak_to_probes(sequence, args.stride)):
        dataset[write_index, probe_index,:] = probe_to_visible_kmers(probe)
    t1 = time.time()
    print("finished probe from peak ", processed_peaks, " processed in ", (t1 - t0), "seconds")
     
    ttv_processed[split] += 1
    processed_peaks += 1

encoded_data.close()

# pickle all the encoder, decoder dicts
with open(os.path.expanduser(os.path.join(args.prefix, args.picklefile)), 'wb') as f:
    data = {'kmer_to_id': kmer_to_id, 'id_to_wc_kmers': id_to_wc_kmers, 'wc_kmer_to_vec': wc_kmer_to_vec}
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
