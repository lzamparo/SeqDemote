import os
import pandas
import random
import numpy as np
from subprocess import call, check_output
# TODO: fix import of from process_flanks import make_flanks
# TODO: fix import of from seq_hdf5 import encode_sequences
import pybedtools
import plotnine as gg
from mizani import breaks


### Grab the atlas, bedgraph files from the Corces ATAC-seq data
os.chdir(os.path.expanduser('~/projects/SeqDemote/data/ATAC/corces_heme'))

### Read atlas .bed file
atlas = pandas.read_csv("peaks/all_celltypes_peak_atlas.bed", sep="\t", header=0, index_col=None, names=["chr", "start", "end"])
celltypes = [l for l in os.listdir('./peaks') if not l.endswith('.bed')]
atlas['peak_len'] = atlas['end'] - atlas['start']

def get_reps_filenames(celltype):
    prefix = os.path.join(os.getcwd(),'peaks',celltype,'MACS2')
    reps = os.listdir(prefix)
    return [os.path.join(prefix,rep) for rep in reps]


### Take a look at the length distribution for peaks:
#limits = (0,1500)
#major_breaks = breaks.mpl_breaks()(limits)
# minor_breaks=breaks.minor_breaks()(major_breaks,limits)
#labels = [str(l) for l in major_breaks]
# gg.scales.scale_x_discrete(breaks = major_breaks, labels = labels, limits=limits) +\

peak_lengths = gg.ggplot(atlas, gg.aes(x='peak_len')) + \
    gg.geoms.geom_histogram(binwidth = 25) + \
    gg.xlim(0,1500) + \
    gg.labels.xlab('Peak lengths (bp)') + \
    gg.labels.ggtitle("Peak lenghts histogram over the atlas") + \
    gg.themes.theme_seaborn()
    
peak_lengths.save(os.path.join(os.path.expanduser("~/projects/SeqDemote/results/diagnostic_plots/ATAC/"),"rebuilt_atlas_length_histogram.pdf"), width = 10)


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

    












