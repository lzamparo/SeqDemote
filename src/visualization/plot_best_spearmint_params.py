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

# plot best values
p = gg.ggplot(gg.aes(y='objective', x='value', fill='param'), df)
p = p + gg.geom_point(show_legend=False) 
p = p + gg.geom_density_2d(show_legend=False)
p = p + gg.facet_wrap('~ param', ncol=3, scales='free_x')
p = p + gg.theme(figure_size=(8,8))
p = p + gg.theme(subplots_adjust={'wspace': 0.5,'hspace': 0.5})
p.save(filename='spearmint_hyperparam_values.png', path=os.path.expanduser("~/projects/SeqDemote/results/BindSpace_embedding_extension/plots"))