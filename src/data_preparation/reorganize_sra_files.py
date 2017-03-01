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

all_sra_files = [f for f in os.listdir('.') if f.endswith('.sra')]

for index, line in experiments.iterrows():
    # extract SRR ID, cell type
    sra_file, cell_type = line['Run_s'], line['source_name_s']
     
    # have we moved this file already?
    if sra_file+".sra" not in all_sra_files:
        continue
    
    # replace spaces to underscores for CD34 types 
    if cell_type in name_to_dirname.keys():
        cell_type = name_to_dirname[cell_type]

    # have we seen this cell type before?
    cell_types = [d for d in os.listdir('.') if os.path.isdir(d)]
    if cell_type not in cell_types: 
        os.mkdir(cell_type)
        cell_type_dict[cell_type] = 1
    else:
        all_reps = [d for d in os.listdir(cell_type) if d.startswith('rep')]
        all_reps.sort()
        last_rep = int(all_reps[-1][-1])
        cell_type_dict[cell_type] = last_rep + 1
        
    # put .sra file into proper replicate structure in cell-type folder
    sra_file_name = sra_file + ".sra"
    rep_suffix = "rep" + str(cell_type_dict[cell_type])
    sra_dump_path = os.path.join(cell_type,rep_suffix)
    os.mkdir(sra_dump_path)
    try:
        os.rename(sra_file_name, os.path.join(sra_dump_path,sra_file_name))
    except:
        print("could not move ", sra_file_name, " to ", os.path.join(sra_dump_path,sra_file_name))
