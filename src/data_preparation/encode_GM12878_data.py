import os
import sys
import pandas
import numpy as np
import pybedtools
import time

from process_flanks import make_flanks
from subprocess import run
from utils import extraction_utils

### Grab all cells within the GM12878 test atlas 
os.chdir(os.path.expanduser("~/projects/SeqDemote/data/ATAC/GM12878"))
hg19_fasta = os.path.expanduser("~/projects/SeqDemote/data/DNase/genomes/hg19.fa")

# load atlas file
atlas = pandas.read_csv("GM12878_annotated_atlas.csv", index_col=None)
if not os.path.exists("fasta_peak_files/"):
    os.mkdir('fasta_peak_files/')
    
# extract sequences for all peaks in the atlas
try:
    completed_process = run("bedtools" + " getfasta -fi ../../genomes/hg19.fa -bed idr_73.bed -s -fo GM12878_peaks.fa", shell=True)
    if completed_process.returncode != 0:
        print("Child was terminated by signal", completed_process.returncode, file=sys.stderr)
    else:
        print("Child returned", completed_process.returncode, file=sys.stderr)
except OSError as e:
    print("Execution failed:", e, file=sys.stderr)


# load the sequences and reformat as data.frame
chrom_list, start_list, end_list, seq_list = extraction_utils.fasta_to_lists(
    "GM12878_peaks.fa")
peaks_dict = {'chrom': chrom_list, 'start': [int(s) for s in start_list], 'end': [int(e) for e in end_list], 'sequence': seq_list}
peaks_df = pandas.DataFrame.from_dict(peaks_dict)

# merge with annotated atlas
atlas_w_seqs = pandas.merge(atlas, peaks_df, how='inner', on= ["chrom", "start", "end"])

# break into sub-peaks, write out to collection of FASTA files, one per chrom
chromosomes = set(chrom_list)

for chromosome in chromosomes:
    with open(os.path.join('fasta_peak_files/', chromosome + "_peaks.fa"),'w') as outfile:
        chrom_peaks = atlas_w_seqs[atlas_w_seqs.chrom == chromosome]
        for i,peak in chrom_peaks.iterrows():
            chrom, start, end = peak['chrom'], peak['start'], peak['end']
            peak_type = peak['annot']
            width = peak['width']
            sequence = peak['sequence']
            
            for subpeak in extraction_utils.peak_to_subpeak_list(chrom, start, end):
                chrom, substart, subend = subpeak
                subsequence = sequence[substart:subend]
                print(extraction_utils.assemble_subpeak_record(subpeak, peak_type), file=outfile)
                print(sequence, file=outfile)
                
         
# read and transform flanks into sub-flank regions
if not os.path.exists("fasta_flank_files/"):
    os.mkdir('fasta_flank_files/')
   
# sample and encode sequence of flanking regions for the atlas
if not os.path.exists('GM12878_flanks.bed'):
    arg_string = "-o GM12878_flanks idr_73.bed"
    my_args = arg_string.split(sep=' ')
    make_flanks(my_args)

# annotate each flank with the annotation of its corresponding peak
with open('GM12878_flanks_annotated.bed','w') as flank_output:
    
    # Make a putative 'flanks' df from the peaks df
    flank_start_list = []
    flank_end_list = []
    flank_chr_list = []
    flank_annot_list = []
    
    for i, peak in atlas.iterrows():
        flank_start_list.append(peak['start'] - 600) 
        flank_end_list.append(peak['start'])
        flank_start_list.append(peak['end'])
        flank_end_list.append(peak['end'] + 600)
        flank_chr_list.append(peak['chrom'])
        flank_chr_list.append(peak['chrom'])
        flank_annot_list.append(peak['annot'])
        flank_annot_list.append(peak['annot'])
            
    flank_dict = {'chrom': flank_chr_list, 'start': flank_start_list, 'end': flank_end_list, 'annot': flank_annot_list}
    putative_flank_df = pandas.DataFrame.from_dict(flank_dict)
    putative_flank_df = putative_flank_df.drop_duplicates()
    flank_df = pandas.read_csv('GM12878_flanks.bed', sep="\t", names=['chrom','start','end','dn1', 'dn2', 'strand','dn3'])
    flank_df = flank_df.drop_duplicates()
    
    # merge with actual flanks df
    annotated_flanks = pandas.merge(flank_df, putative_flank_df,on=['chrom','start','end'])
    annotated_flanks.to_csv(flank_output, sep="\t", index=False, header=False)
    
            
# extract sequences for all peaks in the atlas
try:
    completed_process = run("bedtools" + " getfasta -fi ../../genomes/hg19.fa -bed GM12878_flanks_annotated.bed -s -fo GM12878_flanks.fa", shell=True)
    if completed_process.returncode != 0:
        print("Child was terminated by signal", completed_process.returncode, file=sys.stderr)
    else:
        print("Child returned", completed_process.returncode, file=sys.stderr)
except OSError as e:
    print("Execution failed:", e, file=sys.stderr)

chrom_list, start_list, end_list, seq_list = extraction_utils.fasta_to_lists(
    "GM12878_flanks.fa")
flanks_dict = {'chrom': chrom_list, 'start': [int(s) for s in start_list], 'end': [int(e) for e in end_list], 'sequence': seq_list}
flanks_df = pandas.DataFrame.from_dict(flanks_dict)

# merge with annotated atlas
atlas_w_seqs = pandas.merge(annotated_flanks, flanks_df, how='inner', on= ["chrom", "start", "end"])
atlas_w_seqs['width'] = atlas_w_seqs['end'] - atlas_w_seqs['start']

# break into sub-peaks, write out to collection of FASTA files, one per chrom
chromosomes = set(chrom_list)

for chromosome in chromosomes:
    with open(os.path.join('fasta_flank_files/', chromosome + "_flanks.fa"),'w') as outfile:
        chrom_flanks = atlas_w_seqs[atlas_w_seqs.chrom == chromosome]
        for i, flank in chrom_flanks.iterrows():
            chrom, start, end = flank['chrom'], flank['start'], flank['end']
            flank_type = flank['annot']
            width = flank['width']
            sequence = flank['sequence']
            
            for subflank in extraction_utils.peak_to_subpeak_list(chrom, start, end):
                chrom, substart, subend = subflank
                subsequence = sequence[substart:subend]
                print(extraction_utils.assemble_subpeak_record(subflank, flank_type), file=outfile)
                print(sequence, file=outfile)
