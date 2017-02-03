import os
import ggplot as gg
import numpy as np
from model_etl import pkl_to_df, pkl_to_training_loss_df

### Peaks vs Flanks specific model validation figure: faceted plot of model AUC, training and validation losses.
model_dir = "~/projects/SeqDemote/results/peaks_vs_flanks"
df = pkl_to_df(model_dir)

title_suffix = os.path.basename(model_dir)

faceted_plot = gg.ggplot(df, gg.aes(x='epoch', y='score', color='model')) + \
    gg.geom_line(size=2.0) + \
    gg.xlab('Epoch') + \
    gg.ggtitle('AUC, validation loss, training loss for all Peaks vs Flanks models') + \
    gg.facet_wrap('measure', scales='free_y', ncol=1)
faceted_plot.save(os.path.join(os.path.expanduser('~/projects/SeqDemote/results/diagnostic_plots/'),title_suffix,'pvf_facet_plots.png'), width=12)
# fig, ax = plt.subplots(3, 1, figsize=(3 * 7.5, 3 * 5))

## Plot all models validation AUC, along with SeqGL comparisons
SeqGL_comparisons = [0.9067819,0.8348001,0.8497134,0.8832141,0.8635396,0.8622086]
SeqGL_celltypes = ["CD14","CD19","CD3","CD34","CD56","H1hesc"]
SeqGL_max = max(SeqGL_comparisons)
SeqGL_min = min(SeqGL_comparisons)
SeqGL_mean = np.asarray(SeqGL_comparisons).mean()

df_auc = df[df['measure'] == 'validation AUC']

# TODO: add in a point to represent the one SeqGL_mean point
validation_auc = gg.ggplot(df_auc, gg.aes(x='epoch', y='score', color='model')) + \
    gg.geom_line(size=2.0) + \
    gg.xlab('Epoch') + \
    gg.ylab('Validation AUC') + \
    gg.ggtitle('AUC of each model evaluated on the validation set') + \
    gg.geom_hline(y=SeqGL_mean, size=0.5, color='black',linetype='dotted') + \
    gg.scale_y_continuous(breaks=[.1, .2, .3, .4, .5, .6, .7, .8, SeqGL_mean, .9, 1.0], labels=[.1, .2, .3, .4, .5, .6, .7, .8, 'SeqGL mean', .9, 1.0])
validation_auc.save(os.path.join(os.path.expanduser('~/projects/SeqDemote/results/diagnostic_plots/'),title_suffix,'validation_auc.png'), width=14)

## Plot all models training error
training_loss_df = pkl_to_training_loss_df(model_dir)
training_error = gg.ggplot(training_loss_df, gg.aes(x='chunk', y='score', color='model')) + \
    gg.geom_line(size=2.0) + \
    gg.xlab('Chunk') + \
    gg.ylab('Training loss') + \
    gg.ggtitle('Chunk averaged cross-entropy loss on training set') + \
    gg.ylim(low=0,high=1)
training_error.save(os.path.join(os.path.expanduser('~/projects/SeqDemote/results/diagnostic_plots/'),title_suffix,'training_error.png'), width=14)
