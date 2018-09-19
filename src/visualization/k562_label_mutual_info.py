import os
import h5py
import itertools as it
import numpy as np
import plotnine as gg
import pandas as pd

from sklearn.metrics import normalized_mutual_info_score


# save figure here
save_path = os.path.expanduser("~/projects/SeqDemote/results/diagnostic_plots/BindSpace_embedding")
fig_outpath = os.path.join(save_path,"mutual_info_figure.pdf")
h5_filepath = ps.path.expanduser("~/projects/SeqDemote/data/ATAC/K562/K562_embed_TV_annotated_split.h5")

# get the label data for training
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
	return labels[:,TF_mask_array]
	
# calculate the normalized MI for each pair of labels, put into DF
mi_data = {'TF': [], 'TF ': [], 'Normalized MI': []}
for one, two in it.combinations(TF_list, 2):
	mi_data['TF'].append(one)
	mi_data['TF '].append(two)
	data = get_tf_overlaps(h5f, one, two)
	mi_data['Normalized MI'].append(normalized_mutual_info_score(data[:,0],data[:,1]))	

# plot the df as headmap
df = pd.DataFrame.from_dict(mi_data)
df = df.round(2)

p = gg.ggplot(df, gg.aes('TF', 'TF ', fill='Normalized MI'))
p = p + gg.geom_tile(gg.aes(width=.95, height=.95))
p = p + gg.ggtitle("Normalized mutual information") 
p = p + gg.theme_void()
p = p + gg.theme(plot_background=gg.element_rect(fill='white'))
p = p + gg.theme(axis_text_x=gg.element_text(rotation=90, hjust=0.5, vjust=1))
p = p + gg.theme(axis_text_y=gg.element_text(hjust=1))
p = p + gg.theme(axis_ticks=gg.element_blank())

p.save(fig_outpath)



