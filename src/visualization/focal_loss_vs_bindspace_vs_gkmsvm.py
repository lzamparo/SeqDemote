import os 
import plotnine as gg
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve, f1_score, matthews_corrcoef
from utils import train_utils

factors = ["CEBPB","CEBPG", "CREB3L1", "CTCF","CUX1","ELK1","ETV1",
           "FOXJ2","KLF13","KLF16","MAFK","MAX","MGA","NR2C2",
           "NR2F1","NR2F6","NRF1","PKNOX1","ZNF143"]

### BindSpace full sharing models, limited sharing models, no sharing models, focal loss models 
results_dir = "~/projects/SeqDemote/results/spearmint_runs/pytorch_models/detailed_stats"

### ls-gkmsvm results
gkmsvm_results_dir = "~/projects/SeqDemote/results/gkmsvm_comparison/"

def scores_to_predictions(scores):
    return (scores > 0).astype(np.int)

def gather_gkm_predictions(output_file):
    with open(output_file,'r') as f:
        lines = f.readlines()
    y_hat = np.array([line.split()[-1] for line in lines], dtype=float)
    return y_hat

# read ls-gkmsvm results, read results into tidy df, calc AP, F1, MCC
recall_lvl = 0.5
gkmsvm_dfs = []
exp_root = os.path.expanduser(gkmsvm_results_dir)
for model in os.listdir(exp_root):
    os.chdir(os.path.join(exp_root, model))
    ap_list = []
    f1_list = []
    mcc_list = []
    for f in factors:
        # read peaks, flanks
        peaks_predictions = gather_gkm_predictions(f + "_predicted_peaks.txt")
        flanks_predictions = gather_gkm_predictions(f + "_predicted_flanks.txt")
        y_hat = np.hstack((peaks_predictions, flanks_predictions))
        # calculate AP, store
        peaks_labels = np.ones_like(peaks_predictions)
        flanks_labels = np.zeros_like(flanks_predictions)
        y = np.hstack((peaks_labels, flanks_labels))
        precision, recall, thresholds = precision_recall_curve(y, y_hat)
        idx = (np.abs(recall - recall_lvl)).argmin()  # index of element in recall array closest to recall_lvl
        eff_one = f1_score(y, scores_to_predictions(y_hat))
        mcc = matthews_corrcoef(y, scores_to_predictions(y_hat))
        ap_list.append(precision[idx])
        f1_list.append(eff_one)
        mcc_list.append(mcc)

    gkm_model_name_list = [model for l in range(len(factors))]
    gkmsvm_df = pd.DataFrame.from_dict({'model': gkm_model_name_list, 'factor': factors, 'PR50': ap_list, 'F1': f1_list, 'MCC': mcc_list})
    gkmsvm_dfs.append(gkmsvm_df)

# turn into averaged tidy df
gkmsvm_results = pd.concat(gkmsvm_dfs)
gkmsvm_results_no_factors = gkmsvm_results[['model', 'PR50', 'F1', 'MCC']]
groups = gkmsvm_results_no_factors.groupby('model')
gkmsvm_averaged_results = groups.aggregate(np.mean)
gkmsvm_averaged_results = gkmsvm_averaged_results.reset_index()

gkmsvm_tidy_df = gkmsvm_averaged_results.melt(value_vars=['PR50','F1','MCC'], id_vars='model', var_name='measure', value_name='score')
gkmsvm_tidy_df['trial'] = 40
gkmsvm_tidy_df['score'] = gkmsvm_tidy_df['score'] * 100

# read focal-loss trained bindspace models into tidy df
bindspace_root = os.path.expanduser(results_dir)
dfs = []

for model in os.listdir(bindspace_root):
    try:
        df = pd.read_csv(os.path.join(bindspace_root,model)) 
    except pd.errors.EmptyDataError:
        continue
    dfs.append(df)
    
# read matching bindspace models into tidy df
bindspace_results = pd.concat(dfs)
bindspace_results.rename(columns={' trial': 'trial', ' measure': 'measure', ' score': 'score'}, inplace=True)

# Replace any 'AP' meaures with ''
bindspace_results.loc[bindspace_results['measure'] == 'AP', 'measure'] = "PR50"

# save path
save_path = "/Users/zamparol/projects/SeqDemote/results/BindSpace_embedding_extension/plots"

# Line + dot plot for all measures, aggregated over factor
p = gg.ggplot(gg.aes(y='score', x='trial',color='model'), bindspace_results)
p = p + gg.geom_point()
p = p + gg.ylab('score (x 100)')
p = p + gg.geom_hline(gg.aes(yintercept='score', color='model'), linetype='dashed', size=0.25, show_legend=False, data=gkmsvm_tidy_df)
p = p + gg.ggtitle("Validation measures by model")
p = p + gg.facet_wrap('measure', nrow=3)
p.save('spearmint_model_search_with_gkm_comparison.pdf', path=save_path)



# Plot improvement in AUPR by factor, re-formatting data to long format
# split by factor, find mean & max improvements
#df_by_factor = dfs.groupby('factor')

#def bindspace_by_factor(group):
    #bindspace_aupr = group[group['model'] == 'BindSpace_max']['AP']
    #return np.mean(bindspace_aupr)

#def mean_impr_rel_to_bindspace(group):    
    #bindspace_aupr = group[group['model'] == 'BindSpace_max']['AP']
    #bindspace_aupr = np.mean(bindspace_aupr)
    #group_diff = group['AP'] - bindspace_aupr
    #return np.mean(group_diff)

#def max_impr_rel_to_bindspace(group):    
    #bindspace_aupr = group[group['model'] == 'BindSpace_max']['AP']
    #bindspace_aupr = np.mean(bindspace_aupr)
    #group_diff = group['AP'] - bindspace_aupr
    #return np.max(group_diff)

#def promote_to_df(series, name='AP'):
    #df = series.to_frame()
    #df['factor'] = list(series.index)
    #df = df.rename(columns = {0: name})    
    #return df

#bindspace_baseline = promote_to_df(df_by_factor.apply(bindspace_by_factor), name="BindSpace")
#mean_diffed_by_factor = promote_to_df(df_by_factor.apply(mean_impr_rel_to_bindspace), name="Mean Improvement")
#max_diffed_by_factor = promote_to_df(df_by_factor.apply(max_impr_rel_to_bindspace), name="Max Improvement")
#factor_improvement_df = pd.merge(pd.merge(bindspace_baseline,mean_diffed_by_factor,on='factor'),max_diffed_by_factor,on='factor')
#factor_improvement_df['Mean Improvement'] = factor_improvement_df['Mean Improvement'] + factor_improvement_df['BindSpace']
#factor_improvement_df['Max Improvement'] = factor_improvement_df['Max Improvement'] + factor_improvement_df['BindSpace']

#divider=1.0
#spread_max=1.10

#q = gg.ggplot(factor_improvement_df)
#q = q + gg.geom_segment(gg.aes(x='BindSpace', xend='Max Improvement', y='factor', yend='factor'), size=6, color='#a7a9ac')  
#q = q + gg.geom_point(gg.aes(x="Mean Improvement", y='factor'), size=3, stroke=0.7)
#q = q + gg.ggtitle("Validation AP range by factor")
#q = q + gg.xlab("AP") 
#q = q + gg.ylab("factor")
#q.save('factor_relative_improvement.pdf', path=save_path)

#p = p + gg.geom_vline(xintercept = divider, color='white', size=0.05)
#p = p + gg.geom_text(data=mean_diffed_by_factor, mapping=gg.aes(x=spread_max, y='factor', label='AUPR'), inherit_aes=False, size=12, fontweight='bold', format_string='+{}')

