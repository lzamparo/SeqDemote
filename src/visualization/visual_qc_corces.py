import os
import pandas
import random
import numpy as np
from subprocess import call, check_output
# TODO: fix import of from process_flanks import make_flanks
# TODO: fix import of from seq_hdf5 import encode_sequences
import pybedtools


### Grab all cells within the hematopoetic lineage out of the Roadmap data used for Alvaro's paper

os.chdir(os.path.expanduser('~/projects/SeqDemote/data/ATAC/corces_heme'))

### Read atlas .bed file
atlas = pandas.read_csv("all_celltypes_peak_atlas.bed", sep="\t", header=None, index_col=None, names=["chr", "start", "end"])
celltypes = [l for l in os.listdir('./peaks')]

def get_reps_filenames(celltype):
    prefix = os.path.join(os.getcwd(),'peaks',celltype,'MACS2')
    reps = os.listdir(prefix)
    return [os.path.join(prefix,rep) for rep in reps]


### First test: for a randomly selected peak, get the coverage estimates for all celltypes and all replicates
my_peak = atlas.sample()
bedtool_peak = pybedtools.BedTool(my_peak.to_string(header=False, index=False), from_string=True)

my_type = random.choice(celltypes)
my_bg_filenames = get_reps_filenames(my_type)
my_regions = [bedtool_peak.intersect(pybedtools.BedTool(b), sorted=True) for b in my_bg_filenames]

# map the counts that underlie each intersection, take the average
my_counts = [r.map(pybedtools.BedTool(b), c=4, o='mean') for b in my_bg_filenames for r in my_regions]
print(my_counts[0])










