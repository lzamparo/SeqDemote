from __future__ import print_function
import os

def read_fasta_seq(file_handle):
    ''' generator that reads two lines at a time, yields just the sequence '''
    first = file_handle.readline()
    second = file_handle.readline()
    if not first.startsWith(">"):
        print("data_load_utils.read_fasta_seq_block error: expected a FASTA line, but got sequence instead")
        yield "barf"
    else:
        yield second.strip().upper()
    

def get_test_data_files():
    data_dir = "./test_data"
    return [os.path.join(data_dir, test_file) for test_file in os.listdir("./test_data/") if test_file.endswith(".fa")]


def load_data_from_file(filename, trunc=0):
    test_file = open(filename,'r')
    lines = test_file.readlines()
    test_file.close()
    if trunc == 0:
        return [seq.strip() for seq in lines if not seq.startswith('>')]
    else:
        return [seq.strip()[0:trunc] for seq in lines if not seq.startswith('>')]

