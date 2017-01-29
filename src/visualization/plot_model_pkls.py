import pandas as pd
import ggplot as gg
import numpy as np
import pickle
import sys, os

### produce a diagnostic plot of all models in a specified directory (that exist as .pkl files)
model_dir = sys.argv[1]
try:
    os.chdir(os.path.expanduser(model_dir))
except FileNotFoundError as f:
    print("Wrong directory?  Can't cd to ", model_dir)
    exit(1)

# first, grab the models, turn the .pkl files into objects, and get the data
model_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
model_names = [f.split('.')[0] for f in model_files]
models = [pickle.load(open(f,'rb')) for f in model_files]

# next, turn this collection of python objects into a handful of tidy data pandas dfs
dfs_auc = [pd.DataFrame.from_items([('model',[name for i in range(len(model['losses_valid_auc']))]),('epoch',[i for i in range(len(model['losses_valid_auc']))]),('validation AUC', model['losses_valid_auc'])]) for name, model in zip(model_names, models)]
df_auc = pd.concat(dfs_auc)

dfs_vloss = [pd.DataFrame.from_items([('model',[name for i in range(len(model['losses_valid_xent']))]),('epoch',[i for i in range(len(model['losses_valid_xent']))]),('validation Xent loss', model['losses_valid_xent'])]) for name, model in zip(model_names, models)]
df_vloss = pd.concat(dfs_vloss)

### TODO: have to unpack the list of training losses for each epoch within each model.
dfs_tloss = [pd.DataFrame.from_items([('model',[name for i in range(len(model['losses_train']))]),('epoch',[i for i in range(len(model['losses_train']))]),('training Xent loss', np.mean(model['losses_train']))]) for name, model in zip(model_names, models)]
df_tloss = pd.concat(dfs_tloss)


# Finally, plot the validation auc, validation loss, and training loss of each model.  

#fig, ax = plt.subplots(3, 1, figsize=(3 * 7.5, 3 * 5))

title_suffix = os.path.basename(model_dir)

# Plot all models validation AUC
validation_auc = gg.ggplot(df_auc, gg.aes(x='epoch', y='validation AUC', color='model')) + \
    gg.geom_line(size=2.0) + \
        gg.xlab('Epoch') + \
        gg.ylab('Validation AUC') + \
        gg.ggtitle('AUC of each model evaluated on the validation set')
validation_auc.save(os.path.join(os.path.expanduser('~/projects/SeqDemote/results/diagnostic_plots/'),title_suffix,'validation_auc.png'))

## Plot all models validation error
validation_error = gg.ggplot(df_vloss, gg.aes(x='epoch', y='validation Xent loss', color='model')) + \
    gg.geom_line(size=2.0) + \
    gg.xlab('Epoch') + \
    gg.ylab('Validation loss') + \
    gg.ggtitle('Average cross-entropy loss on validation set')
validation_error.save(os.path.join(os.path.expanduser('~/projects/SeqDemote/results/diagnostic_plots/'),title_suffix,'validation_error.png'))

# Plot all models training error
training_error = gg.ggplot(df_tloss, gg.aes(x='epoch', y='training Xent loss', color='model')) + \
    gg.geom_scatter(size=2.0) + \
    gg.xlab('Epoch') + \
    gg.ylab('Training loss') + \
    gg.ggtitle('Average cross-entropy loss on training set')
training_error.save(os.path.join(os.path.expanduser('~/projects/SeqDemote/results/diagnostic_plots/'),title_suffix,'training_error.png'))
