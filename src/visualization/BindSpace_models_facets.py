import os 
import ggplot as gg
import numpy as np
from model_etl import pkl_to_df

### BindSpace models compared against each other
model_dir = "~/projects/SeqDemote/results/BindSpace_embedding_extension"
df = pkl_to_df(model_dir)
df['model'] = df['model'].apply(lambda x: x.lstrip('BindSpace_'))

title_suffix = os.path.basename(model_dir)
df_auroc = df[df['measure'] == "validation AUROC"]
df_aupr = df[df['measure'] == "validation AUPR"]

BindSpace_avg = 0.31415662596238375

auroc_plot = gg.ggplot(df_auroc, gg.aes(x="epoch", y="score", color="model")) + \
    gg.geom_line(size=1.5) + \
    gg.xlab("Epoch") + \
    gg.ggtitle("Validation Set AUROC") + \
    gg.ylab("AUROC") + \
    gg.scale_x_continuous(breaks=[i for i in range(20)], labels=[i for i in range(20)])
auroc_plot.save(os.path.join(os.path.expanduser("~/projects/SeqDemote/results/diagnostic_plots/BindSpace_embedding/"),"AUROC.png"), width=20)

aupr_plot = gg.ggplot(df_aupr, gg.aes(x="epoch", y="score", color="model")) + \
    gg.geom_line(size=1.5) + \
    gg.xlab("Epoch") + \
    gg.ggtitle("Validation Set AUPR") + \
    gg.ylab("AUPR") + \
    gg.geom_hline(y=BindSpace_avg, size=0.5, color='black',linetype='dotted') + \
    gg.scale_x_continuous(breaks=[i for i in range(20)], labels=[i for i in range(20)])
aupr_plot.save(os.path.join(os.path.expanduser("~/projects/SeqDemote/results/diagnostic_plots/BindSpace_embedding/"),"AUPR.png"), width=20)
