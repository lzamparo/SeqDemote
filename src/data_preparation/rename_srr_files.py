# rename SRR..._?.fastq.gz files to <donor>_<rep>_?.fastq.gz files
# in a way that idendified which files of which cell type are biological 
#   replicates and which are technical replicates
#
#  if files are already sorted into directories by cell-type, then simple 
#  renaming below the replicate level identifies bio replciate & technical 
#  replicate

import os
import pandas
import pdb

# read file identifying SRR to donor, build dicts to identify the map between donor id and SRR number, as well as numbe of replicates seen for each combination of donor, cell type
srr_to_donor_id = {}
donor_cell_type_rep_number = {}

name_to_dirname = {"CD34 Bone Marrow": "CD34_Bone_Marrow", "CD34 Cord Blood": "CD34_Cord_Blood"}

os.chdir(os.path.expanduser("~/projects/SeqDemote/data/ATAC"))
experiments = pandas.read_csv("SraRunTable.txt", sep="\t")

for index, line in experiments.iterrows():
    srr_id, donor, cell_type = line['Run_s'], line['donorid_s'], line['source_name_s']
    srr_to_donor_id[srr_id] = donor

	# replace spaces to underscores for CD34 types 
    if cell_type in name_to_dirname.keys():
        cell_type = name_to_dirname[cell_type]

    key_tuple = (donor, cell_type)

    if key_tuple not in donor_cell_type_rep_number.keys():
    	donor_cell_type_rep_number[key_tuple] = [1]
    else:
    	last_val = donor_cell_type_rep_number[key_tuple][-1]
    	donor_cell_type_rep_number[key_tuple].append(last_val + 1) 


# reverse lists in donor, cell type dict so we can pop replicates
for key in donor_cell_type_rep_number.keys():
    donor_cell_type_rep_number[key].sort(reverse=True)

def encode_prefix(filename, celltype, rep_number=None):
    ''' take a fastq file encoded by its SRR id, re-encode it as 
    <donor id>_rep?_<original suffix>'''
    pdb.set_trace()
    srr_id, suffix = filename.split("_")
    donor = srr_to_donor_id[srr_id]
    if not rep_number:
        rep_number = donor_cell_type_rep_number[(donor, celltype)].pop()
    new_name = celltype + "_" + str(donor) + "_" + str(rep_number) + "_" + suffix 
    return new_name, rep_number

# walk directories of data, renaming files as needed
for dirname, dirs, files in os.walk('/cbio/cllab/nobackup/zamparol/heme_ATAC/data/fastq'):
    if 'rep' in dirname:
        prefix, rep = os.path.split(dirname)
        _, celltype = os.path.split(prefix)
        fastq_files = [f for f in files if f.endswith("fastq.gz")]
        rep_number = None
        for f in fastq_files:
            renamed_file, rep_number = encode_prefix(f, celltype, rep_number)
        pdb.set_trace()
        rep_number = None
        print(dirname, " : ", ",".join(fastq_files))
        print(dirname, " : ", ",".join(renamed_files))
