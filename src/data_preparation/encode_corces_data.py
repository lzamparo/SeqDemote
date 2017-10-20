import os
import sys
import pandas
import numpy as np
from subprocess import call, check_output
from process_flanks import make_flanks
from seq_hdf5 import encode_sequences
import pybedtools

      
### utility functions, maybe refactor them out later?
def extend_peak(start,end, length=60):
    ''' If start,end is less than length, extend them.  If start,end is more than length, cut them. '''
    peak_length = end - start
    discrepancy = length - peak_length
    adjustment = np.abs(discrepancy) // 2
    offset = np.abs(discrepancy) % 2
    new_start = start - np.sign(discrepancy) * adjustment
    new_end = end +  np.sign(discrepancy) *(adjustment + offset)
    assert(new_end - new_start == length)
    return new_start, new_end
    

def peak_to_subpeak_list(chrom,start,end):
    """ Take the given peak, split into a list of subregions that make 
    up the peak """
    num_subpeaks = int(end) - int(start) // 60
    start_list = list(range(start,end,60))
    end_list = start_list[1:] 
    end_list.append(start_list[-1] + 60)
    subpeak_lists = [(chrom,s,e) for s,e in zip(start_list,end_list)]
    return subpeak_lists

def extract_mean_activation(celltype, chrom, start, end):
    """ This is a big one.  For a given genomic locus, calculate the 60bp 
    average coverage for this celltype"""
    # get files in peaks/celltype
    my_peak = '\t'.join([chrom,start,end])
    bedtool_peak = pybedtools.BedTool(my_peak, from_string=True)
    
    my_bg_filenames = get_reps_filenames(celltype)
    my_regions = [bedtool_peak.intersect(pybedtools.BedTool(b), sorted=True) for b in my_bg_filenames]
    
    # map the counts that underlie each intersection, take the average across replicates
    my_counts = [r.map(pybedtools.BedTool(b), c=4, o='mean') for b in my_bg_filenames for r in my_regions]
    
    # marshall counts, regions, cell type data into a df
    my_counts_dfs = [c.to_dataframe() for c in my_counts]    
    # get coverage over all regions for all celltypes
    # take the average coverage & return
    return my_counts_dfs

 
def extract_sequence(chrom,start,end,fasta_file):
    """ Extract the sequence of this (sub) peak 
    default should be: os.path.expanduser("~/projects/data/DNase/genomes/hg19.fa") """
    # extract the sequence from this region with pybedtools
    my_peak = '\t'.join([chrom,start,end])
    bedtool_peak = pybedtools.BedTool(my_peak, from_string=True)
    fasta = pybedtools.example_filename(fasta_file)
    a = a.sequence(fi=fasta)
    #print(open(a.seqfn).read())    
    


def peak_to_activation(peak):
    ''' translate a bedfile peak into a fasta identifier for the peak, 
    and an activation list '''
    peak = peak.strip()
    parts = peak.split('\t')
    chrom = parts[0]
    start = parts[1]
    end = parts[2]
    strand = '+'
    
    
    active_in_peaks_str = [code_to_str[p] for p in activations.split(',')]
    active_in_peaks_array = parse_access_pattern('-'.join(active_in_peaks_str))
    peak_ID = chrom + ":" + start + "-" + end + "(" + strand + ")"
    out_list = [peak_ID]
    out_list.extend([str(e) for e in active_in_peaks_array.tolist()])
    act_line = '\t'.join(out_list)    
    return act_line


### Grab all cells within the hematopoetic lineage out of the FastATAC
### data used from Ryan Corces' paper

os.chdir(os.path.expanduser('~/projects/SeqDemote/data/ATAC/corces_heme'))
hg19_fasta = os.path.expanduser('~/projects/SeqDemote/data/DNase/genomes/hg19.fa')

### Read atlas .bed file
atlas = pandas.read_csv("peaks/all_celltypes_peak_atlas.bed", sep="\t", header=0, index_col=None, names=["chr", "start", "end"])
celltypes = [l for l in os.listdir('./peaks') if not l.endswith('.bed')]
atlas['peak_len'] = atlas['end'] - atlas['start']


### Use pybedtools to extract sequences and activations, formatted as FASTA
if not os.path.exists('fasta_peak_files/'):
    maxsubs = 100000
    for i,peak in atlas.iterrows():
        chrom, start, end = peak['chr'], peak['start'], peak['end']
        for subpeak in peak_to_subpeak_list(chrom,start,end):
            sub_chrom, sub_start, sub_end = subpeak
            celltype_activations = [extract_mean_activation(ct, sub_chrom, sub_start, sub_end) for ct in celltypes]
            sequences = [extract_sequence(sub_chrom, sub_start, sub_end, hg19_fasta)]
            
            # write to file
    
    
    
    
    try:
        retcode = call("bedtools" + " getfasta -fi ./genomes/hg19.fa -bed all_celltypes_peak_atlas_unique.bed -s -fo corces_hematopoetic_peaks.fa", shell=True)
        if retcode < 0:
            print("Child was terminated by signal", -retcode, file=sys.stderr)
        else:
            print("Child returned", retcode, file=sys.stderr)
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)


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