import os 
import sys
import pandas
import re
import numpy as np
import pybedtools
from pyfaidx import Fasta

### Data extraction utils
def extract_chrom_start_end(line, extractor):
    ''' Use the compiled re in extractor to return the 
    chrom, start, end of a fasta header line. If the header contains
    the pipe character, then forget about it '''
    # >chr9:69381395-69381904(*)
    match = extractor.match(line)
    if not match:
        error_msg = "failed to parse FASTA header: " + line
        raise IOError(error_msg)
    return match.group(1), match.group(2), match.group(3)

def fasta_to_lists(filename):
    ''' parse a fasta file, returning a dict of lists for each of the 
    Fasta file components: chrom, start, end, sequence '''
    header_exctractor = re.compile('>?(chr[X|Y|0-9]{1,2})[-:]([0-9]+)-([0-9]+)')
    chrom_list = []
    start_list = []
    end_list = []
    seq_list = []
    peaks = Fasta(filename, 
                 sequence_always_upper=True)
    for record in peaks:
        chrom, start, end = extract_chrom_start_end(record.name, header_exctractor)
        chrom_list.append(chrom)
        start_list.append(start)
        end_list.append(end)
        seq_list.append(peaks[record.name][:].seq)
    
    return chrom_list, start_list, end_list, seq_list

def extend_peak(start,end, length=60):
    ''' If start,end is less than length, extend them.  
    If start,end is more than length, cut them. '''
    peak_length = end - start
    discrepancy = length - peak_length
    adjustment = np.abs(discrepancy) // 2
    offset = np.abs(discrepancy) % 2
    new_start = start - np.sign(discrepancy) * adjustment
    new_end = end +  np.sign(discrepancy) *(adjustment + offset)
    assert(new_end - new_start == length)
    return new_start, new_end

def get_reps_filenames(celltype, prefix=None, suffix='sorted.bdg'):
    """ Return the sorted bedgraph files for eahc replicate of the given celltype """ 
    if prefix is None:
        prefix = os.path.join(os.getcwd(),'peaks',celltype,'MACS2')
    reps = os.listdir(prefix)
    return [os.path.join(prefix,rep) for rep in reps if rep.endswith(suffix)]

def peak_to_subpeak_list(chrom,start,end,size=60):
    """ Take the given peak, split into a list of subregions that make 
    up the peak """
    num_subpeaks = int(end) - int(start) // int(size)
    start_list = list(range(start,end,size))
    end_list = start_list[1:] 
    end_list.append(start_list[-1] + size)
    subpeak_lists = [(chrom,s,e) for s,e in zip(start_list,end_list)]
    return subpeak_lists

def map_counts(filename, region):
    return region.map(pybedtools.BedTool(filename), c=4, o='mean')

def extract_mean_activation_parallel(celltype, chrom, start, end):
    """ For a given genomic locus, calculate the 60bp 
    average coverage for this celltype 
    *** N.B:  need to normalize for library size, not yet done IIRC *** 
    """
    
    my_peak = '\t'.join([chrom,str(start),str(end)])
    bedtool_peak = pybedtools.BedTool(my_peak, from_string=True)
    my_bg_filenames = get_reps_filenames(celltype)
    
    # map the counts that underlie each intersection, take the average across replicates
    my_dfs = []
    with futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(map_counts(filename, bedtool_peak)) for filename in my_bg_filenames]
        
        for future in futures.as_completed(futures):
            my_dfs.append(future.to_dataframe())
            
        counts_df = pandas.concat(my_dfs)
    
    # marshall counts, regions, cell type data into a df
    counts_df = pandas.concat([c.to_dataframe() for c in my_counts])    
    return counts_df["name"].mean()    

def extract_mean_activation(celltype, chrom, start, end):
    """ For a given genomic locus, calculate the 60bp 
    average coverage for this celltype 
    *** N.B:  need to normalize for library size, not yet done IIRC *** 
    """
    
    my_peak = '\t'.join([chrom,str(start),str(end)])
    bedtool_peak = pybedtools.BedTool(my_peak, from_string=True)
    my_bg_filenames = get_reps_filenames(celltype)
    
    # map the counts that underlie each intersection, take the average across replicates
    my_counts = [bedtool_peak.map(pybedtools.BedTool(b), c=4, o='mean') for b in my_bg_filenames]
    
    # marshall counts, regions, cell type data into a df
    counts_df = pandas.concat([c.to_dataframe() for c in my_counts])    
    return counts_df["name"].mean()



def extract_sequence(chrom,start,end,fasta_file):
    """ Extract the sequence of this (sub) peak 
    default should be: os.path.expanduser("~/projects/data/DNase/genomes/hg19.fa") """
    # extract the sequence from this region with pybedtools
    my_peak = '\t'.join([chrom,str(start),str(end)])
    bedtool_peak = pybedtools.BedTool(my_peak, from_string=True)
    fasta = pybedtools.example_filename(fasta_file)
    a = bedtool_peak.sequence(fi=fasta)
    return open(a.seqfn).read()


def assemble_subpeak_record(subpeak, peak_type):
    ''' Assemble the FASTA header of a subpeak '''
    header = '>' + '-'.join([str(s) for s in subpeak]) + '\t' + peak_type
    return header


def assemble_subpeak_record_celltype(subpeak, celltype_activations, sequence):
    """ Assemble the FASTA record of sequence and activation 
    Header gets parsed in make_subpeak_header, activation 
    records in which celltype(s) this peak is active. 
    """
    
    # make the header
    header ='\t'.join([s for s in subpeak])
    header = '>' + header
    
    # make the activation string
    activation = ';'.join([ct+' '+str(score) for (ct,score) in celltype_activations])
    
    # append the sequence
    seq_string = str(sequence)
    return header, seq_string

def write_subpeak_record(subpeak_record, outfile):
    header, activation, seq_string = subpeak_record
    print(header, file=outfile)
    print(activation, file=outfile)
    print(seq_string, file = outfile)

def get_filehandle(filenum, prefix='fasta_peak_files/'):
    filename = 'subpeak_seqs_no_activation_' + str(filenum) + '.fa'
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    return open(os.path.join(prefix,filename), 'w')

def turnover_filehandle(outfile, filenum, prefix='fasta_peak_files/'):
    outfile.close()
    return get_filehandle(filenum,prefix)
    









    
