from __future__ import print_function
import argparse
import os, re
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Epoch #  1   train loss =  38.257, time elapsed per 400 batches was 246.416 s
# valid loss =  44.189, AUC = 0.7388, time elapsed for 400 batches = 68.947932958603 s

# read and scrub training error and validation auc data from given files
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

## Epoch #  1   train loss =  38.246, valid loss =  36.203, AUC = 0.8030, time =  58mDEBUG: wrote model checkpoint to: /root/Basset/data/models/basset_best_check.th best!DEBUG: wrote best model to: /root/Basset/data/models/basset_best_best.thESC[0mESC[0m
    
# read and scrub timing infomation from the given file    
def scrub_timing(filename):   
    training_time_list = []
    timeext = re.compile("([\d]+)m")
    with open(filename, 'r') as tracefile:
            for line in tracefile:
                if line.startswith('Epoch'):
                    training_time_list.append(float(timeext.findall(line.strip())[0])) 
    return training_time_list     
    

## Epoch #  1   train loss =  41.249, time elapsed per 200 batches was 2109.607 s

# read and scrub timing information from the given file
def scrub_timing_cudnn(filename):
    training_time_list = []
    cudnn_training_time_list = []
    
    timeext = re.compile("([\d]+\.[\d]+) s")
    init_matches = 0
    with open(filename, 'r') as tracefile:
            for line in tracefile:
                if line.startswith('Initialized'):
                    init_matches = init_matches + 1
                
                if line.startswith('Epoch') and init_matches == 1:
                    training_time_list.append(float(timeext.findall(line.strip())[0])) 
                if line.startswith('Epoch') and init_matches == 2:
                    cudnn_training_time_list.append(float(timeext.findall(line.strip())[0]))
    return training_time_list, cudnn_training_time_list    
    

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

# plot validation auc for SeqDeep, Basset
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
 
# density plot of per epoch training times for Basset    
def plot_training_time(epoch_times, outdir):
    fig = plt.figure()
    sns.set(style="white", palette="muted", color_codes=True)
    sns.despine(left=True)
    # Plot a historgram and kernel density estimate
    sns.distplot(epoch_times, kde=False, rug=True, color="r")
    plt.title("Training times per epoch")
    plt.ylabel("Epochs")
    plt.xlabel("Time (minutes)")
    plt.savefig(os.path.join(outdir, "training_time.pdf"), format="pdf")    

    
# density plots of per 200 batch trainig times for Basset model; regular conv and cudnn conv
def plot_cudnn_diff_time(training_times, cudnn_training_times, outdir):
    fig = plt.figure()
    sns.set_style("white")
    sns.distplot(training_times, kde=False, rug=True, color='r', label="Regular nn module")
    sns.distplot(cudnn_training_times, kde=False, rug=True, color='b', label="Cudnn module")
    plt.legend()
    sns.despine()
    plt.title("Training times per 200 batches of data")
    plt.ylabel("Epochs")
    plt.xlabel("Time (s)")
    plt.savefig(os.path.join(outdir, "cudnn_vs_nn_time.pdf"), format="pdf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kmerfiles", type=str, default='', help="dir containing the seqdeep kmer trace data")
    parser.add_argument("--seqfiles", type=str, default='', help="dir containing the basset sequence trace data")
    parser.add_argument("--trainingtime", type=str, default='', help="dir containing the timing experiment trace data")
    parser.add_argument("--cudnntime", type=str, default='', help="dir containing the timing experiment trace data for cudnn networks vs nn networks")
    parser.add_argument("--outdir", type=str, help="dir where to put the figs")
    
    args = parser.parse_args()
    
    if args.kmerfiles != '' and args.seqfiles != '': 
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
            
        print("Done...")
        plot_training_error(seqdeep_train, basset_train, args.outdir)
        plot_valid_auc(seqdeep_valid, basset_valid, args.outdir)
        print("Done plotting.")
        
    
    if args.trainingtime != '':
        trace_files = os.listdir(args.trainingtime)
        training_times = []
        print("Found ", len(trace_files), " trace files to parse...")
        print("Parsing trace files to extract training time...")
        for infile in trace_files:
            trace_time = scrub_timing(os.path.join(os.path.expanduser(args.trainingtime),infile))
            print("found ", len(trace_time), " epoch training times.")
            training_times.extend(trace_time)
        print("Done...")
        plot_training_time(training_times, args.outdir)
        print("Done plotting.")
        
    if args.cudnntime != '':
        trace_files = os.listdir(args.cudnntime)
        training_times = []
        cudnn_times = []
        print("Found ", len(trace_files), " trace files to parse...")
        print("Parsing trace files to extract regular and cudnn training times...")
        for infile in trace_files:
            trace_time, cudnn_time = scrub_timing_cudnn(os.path.join(os.path.expanduser(args.cudnntime),infile))
            print("found ", len(trace_time), " regular nn times and ", len(cudnn_time), " cudn times.")
            training_times.extend(trace_time)
            cudnn_times.extend(cudnn_time)
        print("Done...")
        print("Censoring outliers...")
        training_times = [t for t in training_times if t < 100.0]
        plot_cudnn_diff_time(training_times, cudnn_times, args.outdir)
        print("Done plotting.")
        