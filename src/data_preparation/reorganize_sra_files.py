import os
import sys
import pandas


### Grab the table describing the Heme ATAC-seq data sets from Corces et al. organized in .sra files
### extract them into appropriate directories, and prep them for processing by the ATAC-seq pipeline.
cell_type_dict = {}

name_to_dirname = {"CD34 Bone Marrow": "CD34_Bone_Marrow", "CD34 Cord Blood": "CD34_Cord_Blood"}

try:
    os.chdir("/cbio/cllab/nobackup/zamparol/heme_ATAC")
except:
    print("cannot CD to /cbio/cllab/nobackup/zamparol/heme_ATAC.  Are you on Hal?")
    
### Read experiment description file 
experiments = pandas.read_csv("SraRunTable.txt", sep="\t")

for index, line in experiments.iterrows():
    # extract SRR ID, cell type
    sra_file, cell_type = line['Run_s'], line['source_name_s']
    
    # have we seen this cell type before?
    if cell_type not in cell_type_dict.keys():
        os.mkdir(cell_type)
        cell_type_dict[cell_type] = 1
    else:
        cell_type_dict[cell_type] = cell_type_dict[cell_type] + 1
        
    # put .sra file into proper replicate structure in cell-type folder
    sra_file_name = sra_file + ".sra"
    rep_suffix = "rep" + str(cell_type_dict[cell_type])
    if cell_type not in name_to_dirname.keys():
        sra_dump_path = os.path.join([cell_type,rep_suffix])
    else:
        sra_dump_path = os.path.join([name_to_dirname[cell_type],rep_suffix])
    
    os.mkdir(sra_dump_path)
    try:
        os.rename(sra_file_name, os.path.join([sra_path_dump,sra_file_name]))
    except:
        print("could not move ", sra_file_name, " to ", os.path.join([sra_path_dump,sra_file_name]))
