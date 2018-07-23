# coding: utf-8
import os
import h5py
import numpy as np

data_dir = '~/projects/SeqDemote/data/ATAC/K562'
data_dir_prefix = os.path.expanduser(data_dir)

# get file, file handles
old_file = os.path.join(data_dir_prefix, 'K562_embed.h5')
new_file = os.path.join(data_dir_prefix,'K562_embed_TV_annotated_split.h5')
of_handle = h5py.File(old_file,'r')
nf_handle = h5py.File(new_file,'w')

whole_mat = of_handle['matrix'][:]
of_handle.close()

# choose amount of data to partition, get the data into memory
r_size, c_size = whole_mat.shape
p = 0.1
indices = np.random.binomial(1, p, c_size)

# prepare new file for writing
data_group = nf_handle.create_group('/data')
peak_index = data_group.create_dataset('peak_index', indices.shape, dtype='i8')
train_group = nf_handle.create_group('/data/training')
valid_group = nf_handle.create_group('/data/validation')
label_train_group = nf_handle.create_group('/labels/training')
label_valid_group = nf_handle.create_group('/labels/validation')
valid_examples = indices.sum()
train_examples = c_size - valid_examples
train_dset = train_group.create_dataset('train_data', (train_examples,r_size), dtype='f8')
valid_dset = valid_group.create_dataset('valid_data', (valid_examples,r_size), dtype='f8')
print("Choosing ", train_examples, " for training, ", valid_examples, " for validation")

# write the peak ids to the peak_index for cross referencing with fasta files
peak_index[:] = indices[:]

# prepare labels for reading
def parse_line(l):
    l = l.strip()
    parts = l.split()
    return [int(p) for p in parts[1:]]

label_filename = os.path.join(data_dir_prefix, 'K562_atac.label')
label_file = open(label_filename,'r')
dim_labels = label_file.readline().strip().split()
dim_labels = [l.rstrip('"').lstrip('"') for l in dim_labels]

label_data_by_line = [np.array(parse_line(l)) for l in label_file.readlines()]
label_data = np.vstack(tuple(label_data_by_line))
num_labels = label_data.shape[1]
label_file.close()

label_train_dset = label_train_group.create_dataset('train_labels', (train_examples, num_labels), dtype='i8')
label_valid_dset = label_valid_group.create_dataset('valid_labels', (valid_examples, num_labels), dtype='i8')

# label the column names by TF
# access by h5file['dataset'].attrs['column_names']

assert(num_labels == len(dim_labels))
labels_as_array = np.array(tuple(dim_labels)).astype('|S9')
label_train_dset.attrs['column_names'] = labels_as_array
label_valid_dset.attrs['column_names'] = labels_as_array
print("Labeled columns of labels with TF labels")


valid_count = 0    
train_count = 0
print("Writing to structured h5file...")
for n, (flip, label) in enumerate(zip(indices, label_data)):
    if flip == 0:
        train_dset[train_count,:] = whole_mat[:, n] # write the data
        label_train_dset[train_count, :] = label_data[n, :] # write the labels
        train_count += 1
    else:
        valid_dset[valid_count,:] = whole_mat[:, n] # write the data
        label_valid_dset[valid_count, :] = label_data[n, :] # write the labels
        valid_count += 1
    if n % 2000 == 0:
        print("processing datum ", n)
print("Done")        

# Test: are the validation examples the same?
print("Testing to see if validation data is the same between both sets:")
valid_examples_from_nf = valid_dset[:]
valid_examples_from_of = whole_mat[:,indices.astype('bool')]
assert(np.allclose(valid_examples_from_nf,valid_examples_from_of.transpose()))
print("Test passed!")


# close the files
nf_handle.close()
