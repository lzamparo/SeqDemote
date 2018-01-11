import os
import sys
import pandas
import numpy as np
from subprocess import call, check_output
from seq_hdf5 import encode_sequences

### Grab all the cell-specific peaks and flanks for the hematopoetic data
os.chdir(os.path.expanduser('~/projects/SeqDemote/results/SeqGL'))
basedir = os.getcwd()

### Go through all the subdirs, amalgamate the peaks, flanks bed 
my_dirs = os.listdir('.')

### For each dir, produce a set of fasta files, target activation files
for celltype in my_dirs:
    os.chdir(os.path.join(basedir,celltype))
    
    ### Use bedtools to extract fasta formatted sequences based on the bedtools format for both peaks and flanks
    if not os.path.exists(basedir+'training_peaks.fa'):
        try:
            retcode = call("bedtools" + " getfasta -fi ./genomes/hg19.fa -bed " + basedir + "_training_peaks.bed -s -fo " + basedir + "_training_peaks.fa", shell=True)
            if retcode < 0:
                print("Child was terminated by signal", -retcode, file=sys.stderr)
            else:
                print("Child returned", retcode, file=sys.stderr)
            retcode = call("bedtools" + " getfasta -fi ./genomes/hg19.fa -bed " + basedir + "_test_peaks.bed -s -fo " + basedir + "_test_peaks.fa", shell=True)
            if retcode < 0:
                print("Child was terminated by signal", -retcode, file=sys.stderr)
            else:
                print("Child returned", retcode, file=sys.stderr)            
        except OSError as e:
            print("Execution failed:", e, file=sys.stderr)
    
    if not os.path.exists(basedir + 'training_flanks.fa'):
        try:
            retcode = call("bedtools" + " getfasta -fi ./genomes/hg19.fa -bed " + basedir + "_training_flanks.bed -s -fo " + basedir + "_training_flanks.fa", shell=True)
            if retcode < 0:
                print("Child was terminated by signal", -retcode, file=sys.stderr)
            else:
                print("Child returned", retcode, file=sys.stderr)                
            retcode = call("bedtools" + " getfasta -fi ./genomes/hg19.fa -bed " + basedir + "_test_flanks.bed -s -fo " + basedir + "_test_flanks.fa", shell=True)
            if retcode < 0:
                print("Child was terminated by signal", -retcode, file=sys.stderr)
            else:
                print("Child returned", retcode, file=sys.stderr)            
        except OSError as e:
            print("Execution failed:", e, file=sys.stderr)

###  merge the activaiton files, fasta files; 
### TODO: make sure the activation files have a header, pro
            
            
### encode the sequences

#hema_acts = pandas.read_csv("hematopoetic_peaks_act.txt", sep="\t")
#assert(sum(hema_acts['H1hesc']) == 62002) # checks out
#assert(sum(hema_acts['CD34']) == 54766) # short 4 ??
#assert(sum(hema_acts['CD14']) == 48303) # checks out
#assert(sum(hema_acts['CD56']) == 36120) # checks out
#assert(sum(hema_acts['CD3']) == 35188)  # checks out
#assert(sum(hema_acts['CD19']) == 37218) # checks out

### Encode the peaks fasta sequences, activity table in a tensor, store in an h5 file
try:
    peak_arg_string = "-b 256 -s 1024 -t 0.15 -v 0.15 -g peaks hematopoetic_peaks.fa hematopoetic_peaks_act.txt hematopoetic_data.h5"
    my_peak_args = peak_arg_string.split(sep=' ')
    encode_sequences(my_peak_args)
except Exception as e:
    print("Could not encode peaks data:", e, file=sys.stderr)
    

### Encode the flanks fasta sequences, activity table in a tensor, store in an h5 file
try:
    flank_arg_string = "-b 256 -s 1024 -t 0.15 -v 0.15 -g flanks hematopoetic_flanks.fa hematopoetic_flanks_act.txt hematopoetic_data.h5"
    my_flank_args = flank_arg_string.split(sep=' ')
    encode_sequences(my_flank_args)
except Exception as e:
    print("Count not encode flanks data:", e, file=sys.stderr)








