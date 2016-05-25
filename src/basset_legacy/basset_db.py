#!/usr/bin/env python
from optparse import OptionParser
from collections import OrderedDict
import os
import random
import subprocess

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

################################################################################
# basset_db.py
#
# Study database motifs.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <db_file> <model_file> <test_hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='model_hdf5_file', default=None, help='Pre-computed model output as HDF5.')
    parser.add_option('-o', dest='out_dir', default='.')
    parser.add_option('-s', dest='sample', default=256, type='int', help='Sequences to sample [Default: %default]')
    parser.add_option('-t', dest='targets_file', default=None, help='File labelings targets in the second column [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide MEME db file, Basset model file, and test data in HDF5 format.')
    else:
        db_file = args[0]
        model_file = args[1]
        test_hdf5_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #################################################################
    # load data
    #################################################################
    # load sequences
    test_hdf5_in = h5py.File(test_hdf5_file, 'r')
    seq_vecs = np.array(test_hdf5_in['test_in'])
    seq_targets = np.array(test_hdf5_in['test_out'])
    test_hdf5_in.close()

    # read target labels
    if options.targets_file:
        target_labels = [line.split()[1] for line in open(options.targets_file)]
    else:
        target_labels = ['t%d'%(ti+1) for ti in range(seq_targets.shape[1])]

    #################################################################
    # sample
    #################################################################
    if options.sample is not None:
        # choose sampled indexes
        sample_i = np.array(random.sample(xrange(seq_vecs.shape[0]), options.sample))

        # filter
        seq_vecs = seq_vecs[sample_i]
        seq_targets = seq_targets[sample_i]

        # create a new HDF5 file
        sample_hdf5_file = '%s/sample.h5' % options.out_dir
        sample_hdf5_out = h5py.File(sample_hdf5_file, 'w')
        sample_hdf5_out.create_dataset('test_in', data=seq_vecs)
        sample_hdf5_out.close()

        # update test HDF5
        test_hdf5_file = sample_hdf5_file


    #################################################################
    # write motifs to hdf5
    #################################################################
    db_motifs = OrderedDict()
    read_motif = False
    for line in open(db_file):
        a = line.split()
        if len(a) == 0:
            read_motif = False
        else:
            if a[0] == 'MOTIF':
                if a[2][0] == '(':
                    protein = a[2][1:a[2].find(')')]
                else:
                    protein = a[2]

            elif a[0] == 'letter-probability':
                read_motif = True
                db_motifs[protein] = []

            elif read_motif:
                db_motifs[protein].append(np.array([float(p) for p in a]))

    # convert to arrays and transpose
    for protein in db_motifs:
        db_motifs[protein] = np.array(db_motifs[protein]).T

    # write to hdf5
    motifs_hdf5_file = '%s/motifs.h5' % options.out_dir
    motifs_hdf5_out = h5py.File(motifs_hdf5_file, 'w')
    mi = 1
    for protein in db_motifs:
        motifs_hdf5_out.create_dataset(str(mi), data=db_motifs[protein])
        mi += 1
    motifs_hdf5_out.close()


    #################################################################
    # Torch predict
    #################################################################
    if options.model_hdf5_file is None:
        options.model_hdf5_file = '%s/model_out.h5' % options.out_dir
        torch_cmd = 'basset_db_predict.lua %s %s %s %s' % (motifs_hdf5_file, model_file, test_hdf5_file, options.model_hdf5_file)
        print torch_cmd
        subprocess.call(torch_cmd, shell=True)

    # load model output
    model_hdf5_in = h5py.File(options.model_hdf5_file, 'r')
    scores_diffs = np.array(model_hdf5_in['scores'])
    reprs_diffs = []
    l = 1
    while 'reprs%d'%l in model_hdf5_in:
        reprs_diffs.append(np.array(model_hdf5_in['reprs%d'%l]))
        l += 1
    model_hdf5_in.close()


    #################################################################
    # score diffs
    #################################################################
    motif_scores_df = pd.DataFrame(scores_diffs, index=db_motifs.keys(), columns=target_labels)

    # plot heat map
    plt.figure()
    g = sns.clustermap(motif_scores_df, figsize=(9,30))

    for tick in g.ax_heatmap.get_xticklabels():
        tick.set_rotation(-45)
        tick.set_horizontalalignment('left')
        tick.set_fontsize(2.5)
    for tick in g.ax_heatmap.get_yticklabels():
        tick.set_fontsize(2.5)

    plt.savefig('%s/heat_scores.pdf' % options.out_dir)
    plt.close()

    # print table
    table_out = open('%s/table_scores.txt' % options.out_dir, 'w')

    mi = 0
    for protein in db_motifs:
        for ti in range(scores_diffs.shape[1]):
            cols = (protein, ti, scores_diffs[mi,ti])
            print >> table_out, '%-10s  %3d  %5.2f' % cols
        mi += 1

    table_out.close()

    #################################################################
    # filter diffs
    #################################################################
    for l in range(3):
        motif_filters_df = pd.DataFrame(reprs_diffs[l], index=db_motifs.keys())

        # plot heat map
        plt.figure()
        g = sns.clustermap(motif_filters_df, figsize=(13,30))

        for tick in g.ax_heatmap.get_xticklabels():
            tick.set_rotation(-45)
            tick.set_horizontalalignment('left')
            tick.set_fontsize(2.5)
        for tick in g.ax_heatmap.get_yticklabels():
            tick.set_fontsize(2.5)

        plt.savefig('%s/heat_filters%d.pdf' % (options.out_dir,l+1))
        plt.close()

        # print table
        table_out = open('%s/table_filters%d.txt' % (options.out_dir,l+1), 'w')

        mi = 0
        for protein in db_motifs:
            for fi in range(reprs_diffs[l].shape[1]):
                cols = (protein, fi, reprs_diffs[l][mi,fi])
                print >> table_out, '%-10s  %3d  %7.4f' % cols
            mi += 1

        table_out.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
