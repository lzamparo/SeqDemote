#!/usr/bin/env python
from optparse import OptionParser
import h5py

import numpy as np

import dna_io

################################################################################
# hdf5_fasta.py
#
# Extract the FASTA sequences from an HDF5 file.
################################################################################

################################################################################
# main
############################s####################################################
def main():
    usage = 'usage: %prog [options] <hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('-s', dest='set', default='test', help='Set (train/valid/test) [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide Basset model file and input sequences (as a FASTA file or test data in an HDF file')
    else:
        hdf5_file = args[0]

    # load 1 hot coded seqeunces from HDF5
    hdf5_in = h5py.File(hdf5_file, 'r')
    seqs_1hot = np.array(hdf5_in['%s_in' % options.set])
    try:
        seq_headers = np.array(hdf5_in['test_headers'])
    except:
        seq_headers = None
    hdf5_in.close()

    # convert to ACGT sequences
    seqs = dna_io.vecs2dna(seqs_1hot)

    for i, seq in enumerate(seqs):
        if seq_headers is None:
            header = 'seq%d' % i
        else:
            header = seq_headers[i]
        print '>%s\n%s' % (header, seq)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    #pdb.runcall(main)
