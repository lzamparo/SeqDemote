import os
import sys
import pandas
import numpy as np
import pybedtools
import time

from process_flanks import make_flanks
from subprocess import run
from utils import extraction_utils


### Grab all cells within the select GM12878 ChIP-seq factor atlases
os.chdir(os.path.expanduser("~/projects/SeqDemote/data/ChIP/GM12878"))
hg19_fasta = os.path.expanduser("~/projects/SeqDemote/data/genomes/hg19.fa")


# make the peaks & flanks regions for each factor
factors = [f for f in os.listdir(".") if f.endswith("peaks.csv")]


if not os.path.exists("chip_fasta_files/"):
    os.mkdir('chip_fasta_files/')


for f in factors:  
    
    # convert each .csv into a bed file: keep seqnames,start,end,'.',score,"*"
    prefix, celltype, experiment, _ = f.split("_")
    peak_bed = prefix + "_" + celltype + "_" + experiment + ".bed"
    peak_sequences = "peak" + prefix + "_" + celltype + "_" + experiment + ".fa"
    peak_file = prefix + "_" + celltype + "_" + experiment + "_peaks.fa"
    flank_bed = prefix + "_" + celltype + "_"+ experiment + "_flanks.bed"
    flank_sequences = "flank" + prefix + "_" + celltype + "_" + experiment + ".fa"
    flank_file = prefix + "_" + celltype + "_" + experiment + "_flanks.fa"    
    
    csvfile = pandas.read_csv(f)
    csvfile['fakestrand'] = pandas.Series(["." for i in range(csvfile.shape[0])])
    idr_peaks = csvfile[csvfile["score"] > 830]
    
    idr_peaks_nodups = idr_peaks.drop_duplicates(subset=["seqnames","start","end"])
    
    # keep only those peaks with the mode width.  Also no mitochondrial peaks.
    peak_length = idr_peaks.mode().loc[0,"width"].astype(np.int64)
    #idr_peaks['seqnames'] = idr_peaks['seqnames'].astype('|S80')
    idr_peaks_nodups = idr_peaks_nodups[(idr_peaks_nodups["width"] == peak_length) & (idr_peaks_nodups["seqnames"] != "chrM")]
    
    idr_peaks_nodups.to_csv(peak_bed, sep="\t", 
                  columns=["seqnames", "start", "end", "fakestrand", "score", "strand"], header=False, index=False)
    
    # make flank bed from peaks
    arg_string = "-o " + flank_bed + " -s " + str(peak_length) + " " + peak_bed 
    my_args = arg_string.split(sep=' ')
    make_flanks(my_args)    
    
    # eliminate duplicates in the flank sequences 
    
    # extract sequences for all peaks
    try:
        peak_extraction = run("bedtools" + " getfasta -fi ../../genomes/hg19.fa -bed " + peak_bed + " -s -fo " + peak_sequences, shell=True)
        if peak_extraction.returncode != 0:
            print("Peaks: child was terminated by signal", peak_extraction.returncode, file=sys.stderr)
        else:
            print("Peaks: child returned", peak_extraction.returncode, file=sys.stderr)
            
        flank_extraction = run("bedtools" + " getfasta -fi ../../genomes/hg19.fa -bed " + flank_bed + " -s -fo " + flank_sequences, shell=True)
        if flank_extraction.returncode != 0:
            print("Flanks: child was terminated by signal", flank_extraction.returncode, file=sys.stderr)
        else:
            print("Flanks: child returned", flank_extraction.returncode, file=sys.stderr)            
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)


    # load the sequences and reformat as data.frame
    chrom_list, start_list, end_list, seq_list = extraction_utils.fasta_to_lists(
        peak_sequences, "drop")
    peaks_dict = {'chrom': chrom_list, 'start': [int(s) for s in start_list], 'end': [int(e) for e in end_list], 'sequence': seq_list}
    peaks_df = pandas.DataFrame.from_dict(peaks_dict)
    
    chrom_list, start_list, end_list, seq_list = extraction_utils.fasta_to_lists(
        flank_sequences, "drop")
    flanks_dict = {'chrom': chrom_list, 'start': [int(s) for s in start_list], 'end': [int(e) for e in end_list], 'sequence': seq_list}
    flanks_df = pandas.DataFrame.from_dict(flanks_dict)    

    if not os.path.exists(os.path.join("chip_fasta_files/",prefix)):
        os.mkdir(os.path.join("chip_fasta_files/",prefix))
    
    with open(os.path.join('chip_fasta_files/', prefix, peak_file),'w') as outfile:
        for i,peak in peaks_df.iterrows():
            header = extraction_utils.write_fasta_header(peak['chrom'],peak['start'],peak['end'],"ChIP_peak")
            sequence = peak['sequence']      
            print(header, file=outfile)
            print(sequence, file=outfile)
            
    with open(os.path.join('chip_fasta_files/', prefix, flank_file),'w') as outfile:
        for i,flank in flanks_df.iterrows():
            header = extraction_utils.write_fasta_header(flank['chrom'],flank['start'],flank['end'],"ChIP_flank")
            sequence = flank['sequence']      
            print(header, file=outfile)
            print(sequence, file=outfile)    

    # clean up interim files
    os.remove(peak_bed)
    os.remove(peak_sequences)
    os.remove(flank_bed)
    os.remove(flank_sequences)
         

# tidy up all .fai files
for f in [z for z in os.listdir(".") if z.endswith(".fai")]:
    os.remove(f)