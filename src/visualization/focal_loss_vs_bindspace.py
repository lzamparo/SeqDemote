import os 
import plotnine as gg
import numpy as np
import pandas as pd


# read focal-loss trained bindspace models into tidy df
results_dir = "~/projects/SeqDemote/results/spearmint_runs/pytorch_models/detailed_stats"
bindspace_root = os.path.expanduser(results_dir)
dfs = []

for model in [f for f in os.listdir(bindspace_root) if f.startswith("BindSpace_two")]:
    dfs.append(pd.read_csv(os.path.join(bindspace_root,model))) 
    
# read matching bindspace models into tidy df
bindspace_results = pd.concat(dfs)
bindspace_results.rename(columns={' trial': 'trial', ' measure': 'measure', ' score': 'score'}, inplace=True)

# code the focal loss model and corresponding BindSpace model together
bindspace_results['code'] = 'blerg'
model_files = [f for f in os.listdir(bindspace_root) if f.startswith("BindSpace_two")]
models = set([m.split('.py')[0].strip('_focal_loss') for m in model_files])
for c,m in enumerate(models):
    m_fl = m + '_focal_loss'
    bindspace_results.loc[bindspace_results['model'].isin([m,m_fl]),'code'] = str(c)

# save path
save_path = "/Users/zamparol/projects/SeqDemote/results/BindSpace_embedding_extension/plots"

# Line + dot plot for all measures, aggregated over factor
p = gg.ggplot(gg.aes(y='score', x='measure',fill='model'), bindspace_results)
p = p + gg.geom_boxplot()
p = p + gg.ylab('score (x 100)')
p = p + gg.ggtitle("Validation measures by model")
p = p + gg.facet_wrap('code', nrow=2)
p.save('spearmint_model_search_focal_loss.pdf', path=save_path)