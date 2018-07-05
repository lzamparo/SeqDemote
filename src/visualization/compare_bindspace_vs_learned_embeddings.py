import os
import torch
import umap
import plotnine as gg
import numpy as np
import pandas as pd
import importlib.util


TF_overlaps = ["CEBPB","CEBPG", "CREB3L1", "CTCF",
                "CUX1","ELK1","ETV1","FOXJ2","KLF13",
                "KLF16","MAFK","MAX","MGA","NR2C2",
                "NR2F1","NR2F6","NRF1","PKNOX1","ZNF143"]

directory_root = "/Users/zamparol/projects/SeqDemote/"

# Pick a model, load the model, extract the conv weights
model_savename = "BindSpace_two_conv_orthogonal_penalty_fc_l1.py-lt09-20180621-182059.ptm"
model_config = "torch_models/embedded_ATAC_models/BindSpace_two_conv_orthogonal_penalty_fc_l1.py"
model_savepath = os.path.join(directory_root,"results/BindSpace_embedding_extension", model_savename)
model_path_name = os.path.join(os.path.expanduser("~/projects/SeqDemote/src"),'models',model_config)

spec = importlib.util.spec_from_file_location(model_config, model_path_name)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

model = model_module.net
model.load_state_dict(torch.load(model_savepath, map_location=lambda storage, loc: storage))
learned_factors = []
for name, p in model.named_parameters():
    if 'orth' in name and 'weight_v' in name:
        learned_factors = p.squeeze()

# Extract the Tf embeddings from the p
bindspace_path = os.path.join(directory_root,"data/invitro/BindSpace/model_0.2_300_15_0.model.tsv")

# get the TF lines, keep those in our own list
bindspace_factors = []
embedded_bindspace_factors = []

def parse_line(l):
    parts = l.strip().split()
    tf_name = parts[0].lstrip("#TF_")
    return tf_name, parts
    

with open(bindspace_path, 'r') as f:
    for line in f:
        if line.startswith("#TF"):
            name, parts = parse_line(line)
            if name in TF_overlaps:
                bindspace_factors.append(name)
                embedded_bindspace_factors.append(np.array([float(f) for f in parts[1:]]))




# Are any of the convolutional filters aligned in BindSpace with the emedded TFs?
# N.B: 2-norm of embedded bindspace factors is scaled to be ~1.0
factor_neighbours = []
for n, f in zip(range(learned_factors.size(0)),learned_factors):
    f_npy = f.detach().numpy()
    neighbours = []
    dists = []
    for (name,b) in zip(bindspace_factors,embedded_bindspace_factors):
        prod = np.inner(f_npy,b)
        neighbours.append(name)
        dists.append(prod)
    names = [str(n) for l in range(len(neighbours))]
    df = pd.DataFrame.from_dict({'filter': names, 'TF': neighbours, 'dists': dists})
    factor_neighbours.append(df)
    
df = pd.concat(factor_neighbours)

p = gg.ggplot(gg.aes(y='dists', x='filter', color='TF'), df)
p = p + gg.geom_point()  
p = p + gg.ggtitle("Inner product of TF with Factors")
p = p + gg.coord_flip()
p.save(os.path.join(directory_root,"results/BindSpace_embedding_extension/plots", "Filters_TF_label_inner_products.pdf"))        

