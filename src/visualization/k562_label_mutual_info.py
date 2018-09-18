import sys
import h5py
import numpy as np
import plotnine as gg
import pandas as pd
import os
import itertools as it
from sklearn.metrics import normalized_mutual_info_score

# save figure here
fig_outpath = os.path.join(os.path.expanduser(sys.argv[2]),"mutual_info_figure.pdf")

# get the label data for training
h5_filepath = os.path.expanduser(sys.argv[1])
h5f = h5py.File(h5_filepath, 'r', libver='latest', swmr=True)
TF_list = ["CEBPB","CEBPG", "CREB3L1", "CTCF",
           "CUX1","ELK1","ETV1","FOXJ2","KLF13",
           "KLF16","MAFK","MAX","MGA","NR2C2",
           "NR2F1","NR2F6","NRF1","PKNOX1","ZNF143"]

def get_tf_overlaps(h5f, first_tf, second_tf):
	''' return the labels for each pair of TFs '''
	TF_overlaps = [s.encode('utf-8') for s in [first_tf, second_tf]]
	TF_colnames = h5f['/labels/training/train_labels'].attrs['column_names']
	TF_mask_array = np.array([n in TF_overlaps for n in TF_colnames])
	labels = h5f['/labels/training/train_labels'][:]
	return labels[TF_mask_array]
	
# calculate the normalized MI for each pair of labels, put into DF
mi_data = {'TF': [], 'TF ': [], 'MI': []}
for one, two in it.combinations(TF_overlaps, 2)):
	mi_data['TF'].append(one)
	mi_data['TF '].append(two)
	data = get_tf_overlaps(h5f, one, two)
	mi_data['MI'].append(normalized_mutual_info_score(data[:,0],data[:,1]))	

# plot the df as headmap
df = pd.DataFrame.from_dict(mi_data)

p = gg.ggplot(df, gg.aes('factor(TF)', 'factor(TF )', fill='MI'))
p = p + gg.geom_tile(aes(width=.95, height=.95))
p = p + gg.geom_text(gg.aes(label='MutualInformation'), size=10, color=text_color)
p = p + gg.scale_y_discrete(limits=TF[::-1])          # new
p = p + gg.theme(                                         # new
     axis_ticks=gg.element_blank(),
     panel_background=gg.element_rect(fill='white'))

p.save(fig_outpath)



