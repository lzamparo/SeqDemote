from __future__ import print_function
import data_load_utils as utils
import numpy as np


def test_load_from_file():
    test_files = utils.get_test_data_files()
    filtered_files = [f for f in test_files if f.endswith('50.fa')]
    arr = np.arange(len(filtered_files))
    np.random.shuffle(arr)
    test_file = filtered_files[arr[0]]
    test_seqs = utils.load_data_from_file(test_file)
    assert(len(test_seqs) == 50)
    