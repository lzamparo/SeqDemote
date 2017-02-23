import os
import sys
import numpy as np
from subprocess import call, check_output
from seq_hdf5 import encode_sequences

### Grab all peaks and flanks provided by Manu (hematopoetic lineage out of the Roadmap data used for Alvaro's paper), encode in h5 file
hg19_fasta = os.path.expanduser('~/projects/SeqDemote/data/DNase/genomes/hg19.fa')
os.chdir(os.path.expanduser('~/projects/SeqDemote/results/SeqGL'))
cell_types = os.listdir(".")

root_dir = os.getcwd()

def call_bedtools(genome, input_bed, output_fasta):
    try:
        retcode = call("bedtools" + " getfasta -fi " + hg19_fasta + " -bed " + input_bed + " -s -fo " + output_fasta, shell=True)
        if retcode < 0:
            print("Child was terminated by signal", -retcode, file=sys.stderr)
        else:
            print("Child returned", retcode, file=sys.stderr)
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)     

def call_encode_seqs(group, fasta, activations):
    try:
        peak_arg_string = "-b 256 -s 1024 -l 150 -g " + group + " " + fasta + " " + activations + " short_separated_hematopoetic_data.h5"
        my_peak_args = peak_arg_string.split(sep=' ')
        encode_sequences(my_peak_args)
    except Exception as e:
        print("Could not encode peaks data:", e, file=sys.stderr)    

def group_name_from_bedfile_name(bf):
    cell_type, _, feature_type = bf.split("_")
    return cell_type+"/"+feature_type

def fix_bedfiles(bf):
    thefile = open(bf)
    content = thefile.read()              # read entire file into memory
    thefile.close()
    thefile = open(bf, 'w')             
    thefile.write(content.replace("*", "+"))  # write the file with the text substitution
    thefile.close()    

for ct in cell_types:
    
    # make the test and training peaks, flanks .fa files, encode in h5 file
    os.chdir(os.path.join(root_dir, ct))
    bed_files = [f for f in os.listdir('.') if f.endswith('.bed')]
    activations_files = [f for f in os.listdir('.') if f.endswith('.txt')]
    for bf, act in zip(bed_files, activations_files):
        output_fasta = bf.split('.')[0] + ".fa"
        if not os.path.exists(output_fasta):
            fix_bedfiles(bf)
            call_bedtools(hg19_fasta, bf, output_fasta)
        call_encode_seqs(group_name_from_bedfile_name(bf), output_fasta, act)
    

        