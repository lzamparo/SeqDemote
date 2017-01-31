import pandas as pd
import numpy as np
import pickle
import os

### Perform ETL on all models in a specified directory (that exist as .pkl files), transform into tidy formatted DataFrame
def pkl_to_df(model_dir):
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
    dfs_auc = [pd.DataFrame.from_items([('model',[name for i in range(len(model['losses_valid_auc']))]),('epoch',[i for i in range(len(model['losses_valid_auc']))]), ('measure', ['validation AUC' for i in range(len(model['losses_valid_auc']))]),('score', model['losses_valid_auc'])]) for name, model in zip(model_names, models)]
    df_auc = pd.concat(dfs_auc)
    
    dfs_vloss = [pd.DataFrame.from_items([('model',[name for i in range(len(model['losses_valid_xent']))]),('epoch',[i for i in range(len(model['losses_valid_xent']))]),('measure', ['validation Xent loss' for i in range(len(model['losses_valid_xent']))]),('score', model['losses_valid_xent'])]) for name, model in zip(model_names, models)]
    df_vloss = pd.concat(dfs_vloss)
    
    dfs_tloss = [pd.DataFrame.from_items([('model',[name for i in range(len(model['losses_train']))]),('epoch',[i for i in range(len(model['losses_train']))]),('measure',['mean training loss' for i in range(len(model['losses_train']))]),('score', np.asarray(model['losses_train']).mean(axis=1))]) for name, model in zip(model_names, models)]
    df_tloss = pd.concat(dfs_tloss)
    
    # now concat and return all
    return pd.concat([df_auc, df_vloss, df_tloss])


