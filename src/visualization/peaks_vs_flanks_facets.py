import os
import ggplot as gg
import numpy as np
from model_etl import pkl_to_df

### Peaks vs Flanks specific model validation figure: faceted plot of model AUC, training and validation losses.
model_dir = "~/projects/SeqDemote/results/peaks_vs_flanks"
df = pkl_to_df(model_dir)

title_suffix = os.path.basename(model_dir)

faceted_plot = gg.ggplot(df, gg.aes(x='epoch', y='score', color='model')) + \
    gg.geom_line(size=2.0) + \
    gg.xlab('Epoch') + \
    gg.ggtitle('AUC, validation loss, training loss for all Peaks vs Flanks models') + \
    gg.facet_wrap('measure', scales='free_y')
faceted_plot.save(os.path.join(os.path.expanduser('~/projects/SeqDemote/results/diagnostic_plots/'),title_suffix,'pvf_facet_plots.png'))
# fig, ax = plt.subplots(3, 1, figsize=(3 * 7.5, 3 * 5))

## Plot all models validation AUC, along with SeqGL comparisons
SeqGL_comparisons = [0.9067819,0.8348001,0.8497134,0.8832141,0.8635396,0.8622086]
SeqGL_celltypes = ["CD14","CD19","CD3","CD34","CD56","H1hesc"]
SeqGL_max = max(SeqGL_comparisons)
SeqGL_min = min(SeqGL_comparisons)
SeqGL_mean = np.asarray(SeqGL_comparisons).mean()

validation_auc = gg.ggplot(df_auc, gg.aes(x='epoch', y='validation AUC', color='model')) + \
    gg.geom_line(size=2.0) + \
        gg.xlab('Epoch') + \
        gg.ylab('Validation AUC') + \
        gg.ggtitle('AUC of each model evaluated on the validation set')
validation_auc.save(os.path.join(os.path.expanduser('~/projects/SeqDemote/results/diagnostic_plots/'),title_suffix,'validation_auc.png'))

### Plot all models validation error
#validation_error = gg.ggplot(df_vloss, gg.aes(x='epoch', y='validation Xent loss', color='model')) + \
    #gg.geom_line(size=2.0) + \
    #gg.xlab('Epoch') + \
    #gg.ylab('Validation loss') + \
    #gg.ggtitle('Average cross-entropy loss on validation set')
#validation_error.save(os.path.join(os.path.expanduser('~/projects/SeqDemote/results/diagnostic_plots/'),title_suffix,'validation_error.png'))

## Plot all models training error
#training_error = gg.ggplot(df_tloss, gg.aes(x='epoch', y='training Xent loss', color='model')) + \
    #gg.geom_scatter(size=2.0) + \
    #gg.xlab('Epoch') + \
    #gg.ylab('Training loss') + \
    #gg.ggtitle('Average cross-entropy loss on training set')
#training_error.save(os.path.join(os.path.expanduser('~/projects/SeqDemote/results/diagnostic_plots/'),title_suffix,'training_error.png'))
