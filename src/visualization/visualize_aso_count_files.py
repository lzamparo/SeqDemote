import os
import pandas as pd
import numpy as np
import re
from scipy import stats

from plotnine import ggplot, geom_histogram, facet_wrap , geom_abline, aes

### read in the ASO data, and perform goodness of fit checks on count data

os.chdir(os.path.expanduser("~/projects/SeqDemote/data/ATAC/mouse_asa/mapped_reads/CD8_effector"))
count_files = [f for f in os.listdir(".") if f.endswith(".cnt")]

#1:4496102-4497124	14;28
#1:4747707-4748634	46;47
#1:4768321-4768793	16;10
#1:4780054-4780508	6;2

def parse_counts(f):
    ''' Take an open file descriptor, parse the lines, return as dict of lists '''
    chrom_list = []
    start_list = []
    end_list = []
    ref_counts = []
    alt_counts = []
    for l in f.readlines():
        l = l.strip()
        positions, counts = l.split()
        positions_match = positions_re.match(positions)
        counts_match = counts_re.match(counts)
        if positions_match and counts_match:
            chrom, start, end = positions_match.groups()
            ref_count, alt_count = counts_match.groups()
            chrom_list.append(chrom)
            start_list.append(int(start))
            end_list.append(int(end))
            ref_counts.append(int(ref_count))
            alt_counts.append(int(alt_count))
    return {"chrom": chrom_list, "start": start_list, "end": end_list, "ref_counts": ref_counts, "alt_counts": alt_counts}
        
        
positions_re = re.compile("([0-9|XY]+):([\d]+)\-([\d]+)")
counts_re = re.compile("([\d]+);([\d]+)")
reps = []

for f in count_files:
    with open(f,"r") as myfile:
        reps.append(parse_counts(myfile))
        

reps_df_list = [pd.DataFrame.from_dict(r) for r in reps]

# KS-test goodness of fit for Poisson RVs
for df in reps_df_list:
    sample_size = df.shape[0]
    ref_mean = df['ref_counts'].mean()
    alt_mean = df['alt_counts'].mean()    
    ref_poisson = stats.poisson(ref_mean)
    alt_poisson = stats.poisson(alt_mean)
    
    # split into deciles, compute conditional mean, var, ratio
    binned_ref_counts = pd.qcut(df['ref_counts'], 10, duplicates='drop')
    df['cat_ref_counts'] = binned_ref_counts
    grouped = df.groupby('cat_ref_counts')
    grouped_agg = grouped.agg({'ref_counts': lambda x: [np.mean(x), np.var(x), np.mean(x) / np.var(x)], 'alt_counts': lambda x: [np.mean(x), np.var(x), np.mean(x) / np.var(x)]}) 
    
    # split list column into mean, var, ratio, and promote the index to a full column
    grouped_agg[['ref_mean','ref_var','ref_ratio']] = pd.DataFrame(grouped_agg.ref_counts.values.tolist(), index = grouped_agg.index)
    grouped_agg.reset_index(level=grouped_agg.index.names, inplace=True)
    del grouped_agg['ref_counts']
    del grouped_agg['alt_counts']
    
    ref_plot = ggplot(grouped_agg, aes(x='ref_mean', color='factor(cat_ref_counts)')) + \
    geom_abline(intercept=0,slope='ref_ratio') + \
    geom_abline(intercept=0,slope=1,color='blue',linetype='dashed')
    
    ref_results = stats.kstest(df['ref_counts'], ref_poisson.cdf)
    alt_results = stats.kstest(df['alt_counts'], alt_poisson.cdf)
    
    nb_ref_results = stats.kstest(df['ref_counts'], 'nbinom')
    nb_alt_results = stats.kstest(df['alt_counts'], 'nbinom')
    
    



