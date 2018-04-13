import os
import sys
import pandas
import numpy as np
from subprocess import call, check_output
#from process_flanks import make_flanks
#from seq_hdf5 import encode_sequences

from concurrent.futures import ThreadPoolExecutor, as_completed
import pybedtools
import time


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


def get_reps_filenames(celltype):
    """ Return the sorted bedgraph files for eahc replicate of the given celltype """ 
    prefix = os.path.join(os.getcwd(),'peaks',celltype,'MACS2')
    reps = os.listdir(prefix)
    return [os.path.join(prefix,rep) for rep in reps if rep.endswith('sorted.bdg')]



def peak_to_subpeak_list(chrom,start,end):
    """ Take the given peak, split into a list of subregions that make 
    up the peak """
    num_subpeaks = int(end) - int(start) // 60
    start_list = list(range(start,end,60))
    end_list = start_list[1:] 
    end_list.append(start_list[-1] + 60)
    subpeak_lists = [(chrom,s,e) for s,e in zip(start_list,end_list)]
    return subpeak_lists


def map_counts(input_tuple):
    peak_str, filename = input_tuple
    print("peak_str is: ", peak_str)
    print("filename is: ", filename)
    bedtool_peak = pybedtools.BedTool(peak_str, from_string=True)
    counts = bedtool_peak.map(pybedtools.BedTool(filename), c=4, o='mean')
    return counts.to_dataframe()

def extract_mean_activation_parallel(celltype, chrom, start, end):
    """ For a given genomic locus, calculate the 60bp 
    average coverage for this celltype 
    *** N.B:  need to normalize for library size, not yet done IIRC *** 
    """

    print("(parallel) starting celltype mean activation: ", celltype)
    my_peak = '\t'.join([chrom,str(start),str(end)])
    my_bg_filenames = get_reps_filenames(celltype)

    # map the counts that underlie each intersection, take the average across replicates
    my_dfs = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        input_tups = zip([my_peak for i in range(len(my_bg_filenames))], my_bg_filenames)
        futures = [executor.submit(map_counts, i) for i in input_tups]

        for future in as_completed(futures):
            try:
                print("future is: ", future)
                my_dfs.append(future.result())
            except TypeError as e:
                print("Cannot call result() on future for some reason", e)
                print("future is: ", future)

        counts_df = pandas.concat(my_dfs)

    # marshall counts, regions, cell type data into a df    
    print("(parallel) done celltype mean activation: ", celltype)
    return counts_df["name"].mean()    


def extract_mean_activation(celltype, chrom, start, end):
    """ For a given genomic locus, calculate the 60bp 
    average coverage for this celltype 
    *** N.B:  need to normalize for library size, not yet done IIRC *** 
    """

    print("(sequential) starting celltype mean activation: ", celltype)
    my_peak = '\t'.join([chrom,str(start),str(end)])
    bedtool_peak = pybedtools.BedTool(my_peak, from_string=True)
    my_bg_filenames = get_reps_filenames(celltype)

    # map the counts that underlie each intersection, take the average across replicates
    my_counts = [bedtool_peak.map(pybedtools.BedTool(b), c=4, o='mean') for b in my_bg_filenames]

    # marshall counts, regions, cell type data into a df
    counts_df = pandas.concat([c.to_dataframe() for c in my_counts])   
    print("(sequential) done celltype mean activation: ", celltype)
    return counts_df["name"].mean()


def extract_sequence(chrom,start,end,fasta_file):
    """ Extract the sequence of this (sub) peak 
    default should be: os.path.expanduser("~/projects/data/DNase/genomes/hg19.fa") """
    # extract the sequence from this region with pybedtools
    my_peak = '\t'.join([chrom,str(start),str(end)])
    bedtool_peak = pybedtools.BedTool(my_peak, from_string=True)
    fasta = pybedtools.example_filename(fasta_file)
    a = a.sequence(fi=fasta)
    #print(open(a.seqfn).read())    

def assemble_subpeak_record(subpeak, celltype_activations, sequence):
    """ Assemble the FASTA record of sequence and activation """
    # make the header
    header ='\t'.join([s for s in subpeak])
    header = '>' + header

    # make the activation string
    activation = ';'.join([ct+' '+str(score) for (ct,score) in celltype_activations])

    # append the sequence
    seq_string = str(sequence)
    return header, activation, seq_string


def write_subpeak_record(subpeak_record, outfile):
    header, activation, seq_string = subpeak_record
    print(header, file=outfile)
    print(activation, file=outfile)
    print(seq_string, file = outfile)

def get_filehandle(filenum):
    filename = 'subpeak_seqs_with_ativation_' + str(filenum) + '.fa'
    if not os.path.exists('fasta_peak_files/'):
        os.mkdir('fasta_peak_files/')
    return open(os.path.join('fasta_peak_files',filename), 'w')

def turnover_filehandle(outfile):
    outfile.close()
    filenum += 1
    return get_filehandle(filenum)


### Grab all cells within the hematopoetic lineage out of the FastATAC
### data used from Ryan Corces' paper

os.chdir(os.path.expanduser('~/projects/SeqDemote/data/ATAC/corces_heme'))
hg19_fasta = os.path.expanduser('~/projects/SeqDemote/data/DNase/genomes/hg19.fa')

### Read atlas .bed file
atlas = pandas.read_csv("peaks/all_celltypes_peak_atlas.bed", sep="\t", header=0, index_col=None, names=["chr", "start", "end"])
celltypes = [l for l in os.listdir('./peaks') if not l.endswith('.bed')]
atlas['peak_len'] = atlas['end'] - atlas['start']

outfile = open('peaks/all_celltypes_subpeak_atlas.bed','w')
for i,peak in atlas.iterrows():
    chrom,start, end = peak['chr'], peak['start'], peak['end']
    for subpeak in peak_to_subpeak_list(chrom, start, end):
        sub_chr, sub_start, sub_end = subpeak
        subpeak_str = '\t'.join([sub_chr, str(sub_start), str(sub_end)])
        print(subpeak_str, file=outfile)

print("done")

### Test parallel encoding, to make sure we can speed up blah-blah:
#test_peaks = atlas.head(n = 4)
#for i, peak in test_peaks.iterrows():
    #chrom, start, end = peak['chr'], peak['start'], peak['end']
    #for subpeak in peak_to_subpeak_list(chrom,start,end):
        #print("subpeak ", i)
        #sub_chrom, sub_start, sub_end = subpeak
        #t2 = time.time()
        #celltype_activations_parallel = [(ct, extract_mean_activation_parallel(ct, sub_chrom, sub_start, sub_end)) for ct in celltypes]
        #t3 = time.time()
        #print("parallel activations: %f" % (t3 - t2))        
        #t0 = time.time()
        #celltype_activations = [(ct, extract_mean_activation(ct, sub_chrom, sub_start, sub_end)) for ct in celltypes]
        #t1 = time.time()
        #print("sequential activations: %f" % (t1 - t0))
        
        #### Make sure they produce similar results
        #for (sequential_ct, sequential_act), (parallel_ct, parallel_act)  in zip(celltype_activations, celltype_activations_parallel):
            #assert(sequential_ct == parallel_ct)
            #assert(sequential_act == parallel_act)