from __future__ import print_function
import subprocess
import numpy as np


import contextlib,time
@contextlib.contextmanager
def timeit():
    t=time.time()
    yield
    print(time.time()-t,"sec")
    
def get_fasta_records(filename):
    ''' Count the number of fasta sequence records in the given file. '''
    wc_output = subprocess.check_output(['wc', '-l', filename])
    return int(wc_output.strip().split()[0]) / 2

def get_valid_fasta_records(filename):
    valid_records = 0
    for line in open(filename,'r'):
        if line.startswith('>'):
            continue
        if 'N' not in line.upper():
            valid_records = valid_records + 1
    return valid_records

def split_ttv(filename, test_share=0.1, valid_share=0.1, as_counts=True):
    ''' partition the data in filename into the specified amount of training, test, validation data '''     
    
    # count records in fasta file
    num_records = get_valid_fasta_records(filename)
    if not as_counts:
        test_count = int(0.5 + test_share * num_records)
        valid_count = int(0.5 + valid_share * num_records)
    else:
        test_count = test_share
        valid_count = valid_share
    
    train_count = num_records - test_count - valid_count
    
    return int(train_count), int(test_count), int(valid_count)

    
def split_ttv_indices(filename, train_count, test_count, valid_count, parts=10):
    ''' return the indices of the partitions into training, test, validation for the given counts '''
    num_records = get_valid_fasta_records(filename)
    
    # lower limits on number of records to go into each dataset from each chunk
    # of the input fasta file.
    train_lower_indices = list(np.linspace(0, train_count, parts, endpoint=False, dtype=int))
    test_lower_indices = list(np.linspace(0, test_count, parts, endpoint=False, dtype=int))
    valid_lower_indices = list(np.linspace(0, valid_count, parts, endpoint=False, dtype=int))
        
    train_upper_indices = train_lower_indices[1:]
    train_upper_indices.append(train_count)
    test_upper_indices = test_lower_indices[1:]
    test_upper_indices.append(test_count)
    valid_upper_indices = valid_lower_indices[1:]
    valid_upper_indices.append(valid_count)
     
    return zip(train_lower_indices, train_upper_indices), zip(test_lower_indices, test_upper_indices), zip(valid_lower_indices, valid_upper_indices)


def split_ttv_counts(filename, train_count, test_count, valid_count, parts=10):
    ''' return a list of tuples contiaing the number of records for training, test, split in each part 
    For example, if each part needs 100 records and train, test, valid counts are 80,10,10, return (80,10,10). '''
    
    train_indices, test_indices, valid_indices = split_ttv_indices(filename, train_count, test_count, valid_count, parts)
    tt_counts = [(tr[1] - tr[0], te[1] - te[0], va[1] - va[0]) for tr, te, va in zip(train_indices, test_indices, valid_indices)]
    return tt_counts
        
    