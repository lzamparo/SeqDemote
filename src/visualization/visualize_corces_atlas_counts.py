import os
import pandas
import ggplot as gg
import numpy as np

### explore the distribution of counts for the provided activations in Corces et al.

activations = pandas.read_csv(os.path.expanduser("~/projects/SeqDemote/data/ATAC/cqn_normalized_activations_labeled.tsv"), sep='\t')

### turn the activations table into a tidy data set to get an idea of what the range of values are, and then make a faceted density plot
### (one facet per cell-type)


def tidy_up_df(cell_type_df):
    ''' Take a df which is a set of Series identified as donor-#rep-type, output a tidy 
    formatted df which is formatted as type, donor, replicate, value
    '''
    nrows, _ = cell_type_df.shape
    intermediate_df_list = []
    for colname, colvals in cell_type_df.iteritems():
        donor, replicate, celltype = colname.split('-')
        intermediate_df_list.append(pandas.DataFrame.from_dict({'donor': [donor] * nrows, \
                                                                'replicate': [replicate] * nrows, \
                                                                'celltype': [celltype]* nrows, \
                                                                'values': colvals}))
    return pandas.concat(intermediate_df_list)

# reformat the data into tidy-formatted style
cell_types = ['-HSC', '-GMP', '-CLP', '-LMPP', '-MEP', '-MPP', 'CD4', 'CD8', 'NKcell', 'Ery', 'Bcell', 'Mono']
tidy_celltype_dfs = [tidy_up_df(activations.filter(regex=suffix)) for suffix in cell_types]

for cell_type, df in zip(cell_types, tidy_celltype_dfs):
    cell_type = cell_type.lstrip('-')
    print("working on ", cell_type, " ...")
    cell_type_hist = gg.ggplot(df, gg.aes(x='values')) + \
        gg.geom_histogram(binwidth=0.1) + \
        gg.xlab('Normalized peak values') + \
        gg.ylab('Frequency') + \
        gg.ggtitle(cell_type + ' normalized values (aggregated biological & technical reps) ') 
    cell_type_hist.save(os.path.join(os.path.expanduser("~/projects/SeqDemote/results/diagnostic_plots/ATAC/"),cell_type), width = 10)
    print("done.")



