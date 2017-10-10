import os
import sys
import pandas
import numpy as np
from subprocess import call, check_output
from process_flanks import make_flanks
from seq_hdf5 import encode_sequences


### Grab all cells within the hematopoetic lineage out of the Roadmap data used for Alvaro's paper

os.chdir(os.path.expanduser('~/projects/SeqDemote/data/ATAC/corces_heme'))

### Read atlas .bed file
atlas = pandas.read_csv("all_celltypes_peak_atlas.bed", sep="\t", header=None, index_col=None, names=["chr", "start", "end"])
celltypes = [l for l in os.listdir('./peaks')]


### Establish a dictionary to return cell type codes in the form of an np.array
activity_dict = {n: v for v,n in enumerate(celltypes)}

        
### utility functions, maybe refactor them out later?
def extend_peak(start,end, length=600):
    ''' If start,end is less than length, extend them.  If start,end is more than length, cut them. '''
    peak_length = end - start
    discrepancy = length - peak_length
    adjustment = np.abs(discrepancy) // 2
    offset = np.abs(discrepancy) % 2
    new_start = start - np.sign(discrepancy) * adjustment
    new_end = end +  np.sign(discrepancy) *(adjustment + offset)
    assert(new_end - new_start == length)
    return new_start, new_end
    

# reverse the activity dict, so I can parse the peaks bed file and make them into an activation table
code_to_str = {}
for key in activity_dict.keys():
    val = activity_dict[key]
    code_to_str[val] = key

def calculate_activation(chrom,start,end):
    """ This is a big one.  For a given genomic locus, calculate the 50bp average coverage """


def peak_to_activation(peak):
    ''' translate a bedfile peak into a fasta identifier for the peak, and an activation list '''
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

### Convert Alvaro's data set into a bedfile format expected by preprocess_features.py
if not os.path.exists('hematopoetic_peaks.bed'):
    outfile = open('hematopoetic_peaks.bed','w')    
    for index, line in peaks.iterrows():
        print(alvaro_to_bed_format(line), file=outfile)
    
    outfile.close()    


### Use bedtools to extract fasta formatted sequences based on the bedtools format
if not os.path.exists('hematopoetic_peaks.fa'):
    try:
        retcode = call("bedtools" + " getfasta -fi ./genomes/hg19.fa -bed hematopoetic_peaks.bed -s -fo hematopoetic_peaks.fa", shell=True)
        if retcode < 0:
            print("Child was terminated by signal", -retcode, file=sys.stderr)
        else:
            print("Child returned", retcode, file=sys.stderr)
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)


### Make a matched (or near matched) set of flanks for the peaks in hematopoetic_peaks.bed
if not os.path.exists('hematopoetic_flanks.bed'):
    arg_string = "-o hematopoetic_flanks hematopoetic_peaks.bed"
    my_args = arg_string.split(sep=' ')
    make_flanks(my_args)


### Use bedtools again to get the fasta formatted sequences for the flanks
if not os.path.exists('hematopoetic_flanks.fa'):
    try:
        retcode = call("bedtools" + " getfasta -fi ./genomes/hg19.fa -bed hematopoetic_flanks.bed -s -fo hematopoetic_flanks.fa", shell=True)
        if retcode < 0:
            print("Child was terminated by signal", -retcode, file=sys.stderr)
        else:
            print("Child returned", retcode, file=sys.stderr)
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)

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


### Test the activation file:  
#mski1743:DNase zamparol$ cat peaksTable.txt | grep -c -w H1hesc
#62002
#mski1743:DNase zamparol$ cat peaksTable.txt | grep -c -w CD34
#54766
#mski1743:DNase zamparol$ cat peaksTable.txt | grep -c -w CD14
#48303
#mski1743:DNase zamparol$ cat peaksTable.txt | grep -c -w CD56
#36120
#mski1743:DNase zamparol$ cat peaksTable.txt | grep -c -w CD3
#35188
#mski1743:DNase zamparol$ cat peaksTable.txt | grep -c -w CD19
#37218

hema_acts = pandas.read_csv("hematopoetic_peaks_act.txt", sep="\t")
assert(sum(hema_acts['H1hesc']) == 62002) # checks out
#assert(sum(hema_acts['CD34']) == 54766) # short 4 ??
assert(sum(hema_acts['CD14']) == 48303) # checks out
assert(sum(hema_acts['CD56']) == 36120) # checks out
assert(sum(hema_acts['CD3']) == 35188)  # checks out
assert(sum(hema_acts['CD19']) == 37218) # checks out
        
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