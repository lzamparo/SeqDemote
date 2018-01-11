import os
import sys
import pandas
import numpy as np
from process_flanks import make_flanks
#from seq_hdf5 import encode_sequences

from concurrent.futures import ThreadPoolExecutor, as_completed
import pybedtools
import time
from utils import extraction_utils
     
# TODO: refactor this out as is
os.chdir(os.path.expanduser('~/projects/SeqDemote/data/ATAC/corces_heme'))
hg19_fasta = os.path.expanduser('~/projects/SeqDemote/data/DNase/genomes/hg19.fa')
celltypes = [l for l in os.listdir('./peaks') if not l.endswith('.bed')]

### Break up peaks into sub-peaks atlas
#if not os.path.exists('peaks/all_celltypes_subpeak_atlas.bed'):
    #outfile = open('peaks/all_celltypes_subpeak_atlas.bed','w')
    #for i,peak in atlas.iterrows():
        #chrom,start, end = peak['chr'], peak['start'], peak['end']
        #for subpeak in peak_to_subpeak_list(chrom, start, end):
            #print(subpeak, file=outfile)
    #outfile.close()
            
### Use pybedtools to extract sequences and activations, formatted as FASTA
### Read subpeak atlas .bed file
    
### For now: each peak in each celltype should identify the subpeak atlas where it was sourced
### Eventually: each subpeak gets written out with its mean GC-corrected, sizeFactor normalized coverage score per celltype
    
atlas = pandas.read_csv("peaks/all_celltypes_subpeak_atlas.bed", sep="\t", header=0, index_col=None, names=["chr", "start", "end"])

atlas['peak_len'] = atlas['end'] - atlas['start']


if os.path.exists('fasta_peak_files/'):
    maxsubs = 100000
    records = 0
    filenum = 0
    outfile = extraction_utils.get_filehandle(filenum)
    for i,peak in atlas.iterrows():
        chrom, start, end = peak['chr'], peak['start'], peak['end']
        subpeak_record = extraction_utils.extract_sequence(chrom, start, end, hg19_fasta)
        print(subpeak_record, file=outfile)
        records += 1
        
        if records % maxsubs == 0:
            print("processed record: ", records)
            filenum += 1
            outfile = extraction_utils.turnover_filehandle(outfile, filenum)
            
    outfile.close()        
    
    
    

#try:
    #retcode = call("bedtools" + " getfasta -fi ./genomes/hg19.fa -bed all_celltypes_peak_atlas_unique.bed -s -fo corces_hematopoetic_peaks.fa", shell=True)
    #if retcode < 0:
        #print("Child was terminated by signal", -retcode, file=sys.stderr)
    #else:
        #print("Child returned", retcode, file=sys.stderr)
#except OSError as e:
    #print("Execution failed:", e, file=sys.stderr)


### Make a matched (or near matched) set of flanks for the peaks in hematopoetic_peaks.bed
if not os.path.exists('corces_hematopoetic_flanks.bed'):
    arg_string = "-o corces_hematopoetic_flanks all_celltypes_peak_atlas_unique.bed"
    my_args = arg_string.split(sep=' ')
    make_flanks(my_args)


### Use bedtools again to get the fasta formatted sequences for the flanks
if not os.path.exists('corces_hematopoetic_flanks.fa'):
    try:
        retcode = call("bedtools" + " getfasta -fi ./genomes/hg19.fa -bed corces_hematopoetic_flanks.bed -s -fo corces_hematopoetic_flanks.fa", shell=True)
        if retcode < 0:
            print("Child was terminated by signal", -retcode, file=sys.stderr)
        else:
            print("Child returned", retcode, file=sys.stderr)
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)


##### Here's where things get more complicated.  For a given subpeak, I need to find the 
##### average activation over all celltypes for which I have data.  It's not critical until
##### I need to start learning the embedding, but I do have to deal with it eventually.

### Make the activity table for the peaks by parsing Alvaro's file again
if not os.path.exists('hematopoetic_peaks_act.txt'):
    header = "\t".join(["peakID","H1hesc","CD34","CD14","CD56","CD3","CD19"])
    outfile = open('hematopoetic_peaks_act.txt','w')
    print(header, file=outfile)
    outfile.close()
    with open('hematopoetic_peaks.bed','r') as h_peaks, open('hematopoetic_peaks_act.txt','a') as outfile:
        for line in h_peaks:
            act_line = peak_to_activation(line)
            print(act_line, file=outfile)

        
### Make the activity table for the flanks
if not os.path.exists('hematopoetic_flanks_act.txt'):
    header = "\t".join(["flankID","H1hesc","CD34","CD14","CD56","CD3","CD19"])
    outfile = open('hematopoetic_flanks_act.txt','w')
    print(header, file=outfile)
    outfile.close()
    with open('hematopoetic_flanks.bed','r') as h_flanks, open('hematopoetic_flanks_act.txt','a') as outfile:
        for line in h_flanks:
            line = line.strip()
            parts = line.split('\t')
            flank_ID = parts[0] + ":" + parts[1] + "-" + parts[2] + "(+)"
            act_line = "\t".join([flank_ID,"0","0","0","0","0","0"])
            print(act_line, file=outfile)


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