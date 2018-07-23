import os 
import plotnine as gg
import numpy as np
import pandas as pd

from model_etl import o_to_df

model_dir = "~/projects/SeqDemote/results/spearmint_runs/pytorch_models"
df = o_to_df(model_dir, model_regex='[\d]+.stdout', lstrip='BindSpace_')

### validation AP plot for spearmint trials
p = gg.ggplot(gg.aes(x='trial', y='score', color='model'), df)
p = p + gg.geom_line(size=1.0)
p = p + gg.xlab('Spearmint trial')
p = p + gg.ylab('Validation Average Precision')
p = p + gg.ggtitle('Hyperparameter tuning results for sparsity, convolution penalties')
p.save('spearmint_trials_factors_all.pdf')

