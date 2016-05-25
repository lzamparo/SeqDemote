#!/usr/bin/env python
from optparse import OptionParser
import h5py
import string

import numpy as np

import dna_io

################################################################################
# motif_fasta.py
#
# Filter a FASTA file for only sequences that have a given motif.
################################################################################

################################################################################
# main
############################s####################################################
def main():
    usage = 'usage: %prog [options] <fasta_file> <motif>>'
    parser = OptionParser(usage)
    parser.add_option('-c', dest='center_nt', default=None, type='int', help='Search only the center nucleotides of each sequence [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide FASTA file and motif')
    else:
        fasta_file = args[0]
        motif = args[1]

    motif_rc = rc(motif)

    seq = ''
    for line in open(fasta_file):
        if line[0] == '>':
            if seq:
                print_motif(header, seq, [motif,motif_rc], options.center_nt)
            header = line[1:].rstrip()
            seq = ''
        else:
            seq += line.rstrip()

    if seq:
        print_motif(header, seq, [motif,motif_rc], options.center_nt)


def print_motif(header, seq, motifs, center_nt):
    ''' Print the sequence as FASTA if one of the motifs occurs in the center '''

    if center_nt is None:
        search_seq = seq
    else:
        seq_len = len(seq)
        sides_len = max(0, seq_len - center_nt)
        seq_start = sides_len/2
        seq_end = seq_start + center_nt
        search_seq = seq[seq_start:seq_end]

    hit = False
    for motif in motifs:
        if search_seq.find(motif) != -1:
            hit = True
            break

    if hit:
        print '>%s\n%s' % (header,seq)


def rc(seq):
    ''' Reverse complement sequence'''
    return seq.translate(string.maketrans("ATCGatcg","TAGCtagc"))[::-1]

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    #pdb.runcall(main)
