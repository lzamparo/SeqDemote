import os
import pandas
import random
import numpy as np
from subprocess import call, check_output
# TODO: fix import of from process_flanks import make_flanks
# TODO: fix import of from seq_hdf5 import encode_sequences
import pybedtools
import ggplot as gg


### Grab the atlas, bedgraph files from the Corces ATAC-seq data
os.chdir(os.path.expanduser('~/projects/SeqDemote/data/ATAC/corces_heme'))

### Read atlas .bed file
atlas = pandas.read_csv("all_celltypes_peak_atlas_unique.bed", sep="\t", header=0, index_col=None, names=["chr", "start", "end", "length"])
celltypes = [l for l in os.listdir('./peaks')]

def get_reps_filenames(celltype):
    prefix = os.path.join(os.getcwd(),'peaks',celltype,'MACS2')
    reps = os.listdir(prefix)
    return [os.path.join(prefix,rep) for rep in reps]


### Take a look at the length distribution for peaks:
#peak_lengths = gg.ggplot(atlas, gg.aes(x='length')) + \
    #gg.geom_histogram(binwidth = 200) + \
    #gg.xlab('Peak lengths (bp)') + \
    #gg.ggtitle("Peak lenghts histogram over the atlas")
#peak_lengths.save(os.path.join("~/projects/SeqDemote/results/diagnostic_plots/ATAC/","atlas_length_histogram"), width = 10)


### First test: for a randomly selected peak, get the coverage estimates for all celltypes and all replicates
my_peak = atlas.sample()
bedtool_peak = pybedtools.BedTool(my_peak.to_string(header=False, index=False), from_string=True)

my_type = random.choice(celltypes)
my_bg_filenames = get_reps_filenames(my_type)
my_regions = [bedtool_peak.intersect(pybedtools.BedTool(b), sorted=True) for b in my_bg_filenames]

# map the counts that underlie each intersection, take the average across replicates
my_counts = [r.map(pybedtools.BedTool(b), c=4, o='mean') for b in my_bg_filenames for r in my_regions]

# marshall counts, regions, cell type data into a df
my_regions_dfs = [r.to_dataframe() for r in my_regions]
my_counts_dfs = [c.to_dataframe() for c in my_counts]

# plot the counts bedgraph for the region



for c in my_counts:
    print(c)
    












