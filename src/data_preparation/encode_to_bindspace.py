import os
import argparse
import fnmatch
import pickle
import h5py
import numpy as np

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
    a list of the '''
    
    return [peak[i*stride : (i*stride)+20] for i in range(0, (len(peak) - 20) // stride)]

def probe_to_kmers(probe, kmer_len=8):
    ''' find all kmers in the given probe, including the wildcard matching kmers
    return the integer codes for all matches as a list. '''
    
    kmers = [probe[i:i+kmer_len] for i in range(0, len(probe) - kmer_len + 1, 1)]
    matching_bindspace_elements = []
    for kmer in kmers:
        for element in kmer_to_id.keys():
            if fnmatch.fnmatch(kmer, element):
                matching_bindspace_elements.append(element)
    return [kmer_to_id[element] for element in matching_bindspace_elements] 
    
    
# treat this number of consecutive bases of ATAC-seq peak as a probe to be embedded
probe_len = 20

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", dest='prefix', type=str, help='prefix for where the infile, outfile, pickle should go')
parser.add_argument("--infile", dest='bindspace', type=str, help='bindspace input file')
parser.add_argument("--outfile", dest='h5', type=str, help='h5 output file')
parser.add_argument("--fasta", dest='fasta', type=str, help='input fasta file for ATAC-seq')
parser.add_argument("--labels", dest='labels', type=str, help='label data for ATAC-seq peaks')
parser.add_argument("--picklefile", dest='picklefile', type=str, help='pickle file to store dicts')
parser.add_argument("--stride", dest='stride', type=int, default=1, help='sample probes from peaks using the given stride (default: 1)')
args = parser.parse_args()

# parse the code file, generate both kmer -> int, int -> code dicts
kmer_to_id = OrderedDict()
id_to_vec = OrderedDict()

with open(os.path.expanduser(os.path.join(args.prefix, args.bindspace)),'r') as f:
    for i, line in enumerate(f):
        line = line.strip()
        parts = line.split('\t')
        kmer = parts[0].replace("N","?")
        vec = np.asarray(parts[1:], dtype=np.float)
        kmer_to_id[kmer] = i
        id_to_vec[i] = vec
        
# generate hdf5 file with training, validation, test split peaks
encoded_data = h5py.File(os.path.join(args.prefix, args.h5), 'w',swmr=True)
train_group = encoded_data.create_group("/train")
valid_group = encoded_data.create_group("/valid")
test_group = encoded_data.create_group("/test")


# get fasta records, labels
records = []
with open(os.path.expanduser(os.path.join(args.prefix, args.fasta)), 'r') as f:
    for record in _generate_fasta(f):
        records.append(record)

labels = []
with open(os.path.expanduser(os.path.join(args.prefix, args.labels)), 'r') as f:
    header = f.readline()
    for line in f:
        labels.append(parse_line(line))

assert(len(records) == len(labels))
num_peaks = len(records)

# calculate training / test / validation split
# 0: train, 1: validation, 2: test
train_valid_test_split = np.random.choice([0,1,2], num_peaks, p=[0.8,0.1,0.1]) 
tvs_split_dict = {0: train_group, 1: valid_group, 2: test_group}

# for each peak, choose a split, partition into probes, 
# determine the kmer spectrum of the probes, write 
# their ID as an array.
# - each probe will be its own dataset
# - each peak is its own group within the train/test/valid groups
# this resolves the problem of different lengths of sequence

for record, labels, split in zip(records, labels, train_valid_test_split):
    # make the sub-group within the appropriate split
    peak_id, sequence = record
    peak_group = tvs_split_dict[split]
    peak_group.create_group(peak_id)
    
    # process each probe in the peak
    
    # 
    


