from __future__ import print_function
import argparse
import os, re
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Epoch #  1   train loss =  38.257, time elapsed per 400 batches was 246.416 s
# valid loss =  44.189, AUC = 0.7388, time elapsed for 400 batches = 68.947932958603 s

# read and scrub data from given files
def scrub_file(filename):
    train_list = []
    valid_list = []
    
    trainext = re.compile('([\d]+\.[\d]+),')
    with open(filename, 'r') as tracefile:
        for line in tracefile:
            if line.startswith('Epoch'):
                train_list.append(float(trainext.findall(line.strip())[0])) 
                valid_list.append(float(trainext.findall(line.strip())[-1]))
    return train_list, valid_list            
    

# plot training error & auc versus epoch for both sources
def plot_training_error(kmer_train, seq_train, outdir):
    fig = plt.figure()
    sns.set_style("white")
    plt.plot(kmer_train, label="SeqDeep")
    plt.plot(seq_train, label="Basset")
    plt.legend()
    sns.despine()
    plt.title("Training error")
    plt.xlabel("Epoch")
    plt.ylabel("Training error")
    plt.savefig(os.path.join(outdir, 'train_error.pdf'), format="pdf")  

def plot_valid_auc(kmer_valid, seq_valid, outdir):
    fig = plt.figure()
    sns.set_style("white")
    plt.plot(kmer_valid, label="SeqDeep")
    plt.plot(seq_valid, label="Basset")
    plt.legend()
    sns.despine()
    plt.title("Validation set AUC")
    plt.ylabel("AUC")
    plt.xlabel("Epoch")
    plt.savefig(os.path.join(outdir, "valid_auc.pdf"), format="pdf")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kmerfiles", type=str, help="dir containing the seqdeep kmer trace data")
    parser.add_argument("--seqfiles", type=str, help="dir containing the basset sequence trace data")
    parser.add_argument("--outdir", type=str, help="dir where to put the figs")
    
    args = parser.parse_args()
    seqdeep_trace_files = os.listdir(args.kmerfiles)
    basset_trace_files = os.listdir(args.seqfiles)
    
    seqdeep_train = []
    seqdeep_valid = []
    basset_train = []
    basset_valid = []
    print("Parsing SeqDeep trace files...")
    for infile in seqdeep_trace_files:
        kmer_train, kmer_valid = scrub_file(os.path.join(os.path.expanduser(args.kmerfiles),infile))
        print("found ", len(kmer_train), " training and ", len(kmer_valid), " validation records.")
        seqdeep_train.extend(kmer_train)
        seqdeep_valid.extend(kmer_valid)
    
    print("Parsing Basset trace files...")
    for infile in basset_trace_files:
        seq_train, seq_valid = scrub_file(os.path.join(os.path.expanduser(args.seqfiles),infile))
        print("found ", len(seq_train), " training and ", len(seq_valid), " validation records.")
        basset_train.extend(seq_train)
        basset_valid.extend(seq_valid)        
        
    
    plot_training_error(seqdeep_train, basset_train, args.outdir)
    plot_valid_auc(seqdeep_valid, basset_valid, args.outdir)
    