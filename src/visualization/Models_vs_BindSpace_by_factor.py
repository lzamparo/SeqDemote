import os 
import plotnine as gg
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

factors = ["CEBPB","CEBPG", "CREB3L1", "CTCF","CUX1","ELK1","ETV1",
           "FOXJ2","KLF13","KLF16","MAFK","MAX","MGA","NR2C2",
           "NR2F1","NR2F6","NRF1","PKNOX1","ZNF143"]

### BindSpace models compared against each other
model_dir = "~/projects/SeqDemote/results/BindSpace_embedding_extension/results_per_factor"

### ls-gkmsvm results
gkmsvm_results_dir = "~/projects/SeqDemote/results/gkmsvm_comparison"

def gather_gkm_predictions(output_file):
    with open(output_file,'r') as f:
        lines = f.readlines()
    y_hat = np.array([line.split()[-1] for line in lines], dtype=float)
    return y_hat

# read ls-gkmsvm results, calc AP, read results into tidy df
os.chdir(os.path.expanduser(gkmsvm_results_dir))
ap_list = []
for f in factors:
    # read peaks, flanks
    peaks_predictions = gather_gkm_predictions(f + "_predicted_peaks.txt")
    flanks_predictions = gather_gkm_predictions(f + "_predicted_flanks.txt")
    y_hat = np.hstack((peaks_predictions, flanks_predictions))
    # calculate AP, store
    peaks_labels = np.ones_like(peaks_predictions)
    flanks_labels = np.zeros_like(flanks_predictions)
    y = np.hstack((peaks_labels, flanks_labels))
    ap = average_precision_score(y, y_hat)
    ap_list.append(ap)
    
gkm_model_name_list = ['gkmsvm' for l in range(len(factors))]
gkmsvm_df = pd.DataFrame.from_dict({'model': gkm_model_name_list, 'factor': factors, 'AP': ap_list})

# read bindspace results into tidy df
os.chdir(os.path.expanduser(model_dir))
results = open('bindspace_val_aupr.txt','r').readlines()
results = [l.strip() for l in results]
factor_list, aupr_list = [r.split()[0].lstrip('"').rstrip('"') for r in results], [float(r.split()[-1]) for r in results]
bindspace_model_name_list = ["BindSpace_max" for l in range(len(factor_list))]
bindspace_df = pd.DataFrame.from_dict({'model': bindspace_model_name_list, 'factor': factor_list, 'AP': aupr_list})

#BindSpace_avg = 0.31415662596238375

# read results text files into tidy df
df_list = []
for f in [f for f in os.listdir('.') if f.startswith('one') or f.startswith('two')]:
    results = open(f,'r').readlines()
    results = [l.strip() for l in results]
    aupr_list = [float(r) for r in results]
    model_name_list = [f.rstrip('.txt') for l in range(len(aupr_list))]
    df_list.append(pd.DataFrame.from_dict({'model': model_name_list, 'factor': factors, 'AP': aupr_list}))     

df_list.append(bindspace_df)
df_list.append(gkmsvm_df)
dfs = pd.concat(df_list)
    
# Plot with flipped axes, and calculate the difference for each factor

p = gg.ggplot(gg.aes(y='AP', x='factor',color='model'), dfs)
p = p + gg.geom_point()  
p = p + gg.ggtitle("Validation AP by model, factor")
p = p + gg.coord_flip()
p.save('individual_factor_comparison.pdf')


# Plot improvement in AUPR by factor, re-formatting data to long format
# split by factor, find mean & max improvements
df_by_factor = dfs.groupby('factor')

def bindspace_by_factor(group):
    bindspace_aupr = group[group['model'] == 'BindSpace_max']['AP']
    return np.mean(bindspace_aupr)

def mean_impr_rel_to_bindspace(group):    
    bindspace_aupr = group[group['model'] == 'BindSpace_max']['AP']
    bindspace_aupr = np.mean(bindspace_aupr)
    group_diff = group['AP'] - bindspace_aupr
    return np.mean(group_diff)

def max_impr_rel_to_bindspace(group):    
    bindspace_aupr = group[group['model'] == 'BindSpace_max']['AP']
    bindspace_aupr = np.mean(bindspace_aupr)
    group_diff = group['AP'] - bindspace_aupr
    return np.max(group_diff)

def promote_to_df(series, name='AP'):
    df = series.to_frame()
    df['factor'] = list(series.index)
    df = df.rename(columns = {0: name})    
    return df

bindspace_baseline = promote_to_df(df_by_factor.apply(bindspace_by_factor), name="BindSpace")
mean_diffed_by_factor = promote_to_df(df_by_factor.apply(mean_impr_rel_to_bindspace), name="Mean Improvement")
max_diffed_by_factor = promote_to_df(df_by_factor.apply(max_impr_rel_to_bindspace), name="Max Improvement")
factor_improvement_df = pd.merge(pd.merge(bindspace_baseline,mean_diffed_by_factor,on='factor'),max_diffed_by_factor,on='factor')
factor_improvement_df['Mean Improvement'] = factor_improvement_df['Mean Improvement'] + factor_improvement_df['BindSpace']
factor_improvement_df['Max Improvement'] = factor_improvement_df['Max Improvement'] + factor_improvement_df['BindSpace']

divider=1.0
spread_max=1.10

q = gg.ggplot(factor_improvement_df)
q = q + gg.geom_segment(gg.aes(x='BindSpace', xend='Max Improvement', y='factor', yend='factor'), size=6, color='#a7a9ac')  
q = q + gg.geom_point(gg.aes(x="Mean Improvement", y='factor'), size=3, stroke=0.7)
q = q + gg.ggtitle("Validation AP range by factor")
q = q + gg.xlab("AP") 
q = q + gg.ylab("factor")
q.save('factor_relative_improvement.pdf')

#p = p + gg.geom_vline(xintercept = divider, color='white', size=0.05)
#p = p + gg.geom_text(data=mean_diffed_by_factor, mapping=gg.aes(x=spread_max, y='factor', label='AUPR'), inherit_aes=False, size=12, fontweight='bold', format_string='+{}')

