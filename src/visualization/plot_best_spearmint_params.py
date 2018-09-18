import os
import pandas as pd
import numpy as np
import plotnine as gg
from model_etl import parse_spearmint_params_output

# read the log files
logfiles_dir = os.path.expanduser("~/projects/SeqDemote/results/spearmint_runs/pytorch_models/logfiles")
models = [os.path.join(logfiles_dir,m) for m in os.listdir(logfiles_dir)]
parsed_dfs = [parse_spearmint_params_output(m) for m in models]
df = pd.concat([p for p in parsed_dfs if p is not None])

# plot the best focal loss values and first filters
df_focal = df.loc[df['param'].isin(['alpha', 'gamma', 'first_filters'])]

p = gg.ggplot(gg.aes(y='objective', x='value', fill='param'), df_focal)
p = p + gg.geom_point(show_legend=False) 
p = p + gg.geom_density_2d(show_legend=False)
p = p + gg.facet_wrap('~ param', nrow=1, scales='free_x')
p = p + gg.theme(figure_size=(6,4))
p = p + gg.theme(panel_spacing_y=0.5)
p.save(filename='spearmint_hyperparam_focal_loss_values.pdf', path=os.path.expanduser("~/projects/SeqDemote/results/BindSpace_embedding_extension/plots"))

# plot the best regularization params for the 
df_lambdas = df.loc[df['param'].isin(['bias_lambda', 'orth_lambda', 'sparse_lambda', 'weight_lambda'])]

p = gg.ggplot(gg.aes(y='objective', x='value', fill='param'), df_lambdas)
p = p + gg.geom_point(show_legend=False) 
p = p + gg.geom_density_2d(show_legend=False)
p = p + gg.facet_wrap('~ param', ncol=2, scales='free_x')
p = p + gg.theme(figure_size=(6,6))
p = p + gg.theme(panel_spacing_y=0.5)
p.save(filename='spearmint_hyperparam_lambda_values.pdf', path=os.path.expanduser("~/projects/SeqDemote/results/BindSpace_embedding_extension/plots"))