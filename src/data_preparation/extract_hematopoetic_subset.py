import os
import pandas
import numpy as np

### Grab all cells within the hematopoetic lineage out of the Roadmap data used for Alvaro's paper

os.chdir(os.path.expanduser('~/projects/SeqDemote/data/DNase'))


# read both peaks data file, DNase counts file
peaks = pandas.read_csv("peaksTable.txt", sep="\t")
counts = pandas.read_csv("DNaseCnts.txt", sep="\t")

# establish a dictionary to return indicators in the form of an np.array
pattern_dict = {}
pattern_dict['H1hesc'] = np.asarray([1,0,0,0,0,0], dtype=int)
pattern_dict['CD34'] = np.asarray([0,1,0,0,0,0], dtype=int)
pattern_dict['CD14'] = np.asarray([0,0,1,0,0,0], dtype=int)
pattern_dict['CD56'] = np.asarray([0,0,0,1,0,0], dtype=int)
pattern_dict['CD3'] = np.asarray([0,0,0,0,1,0], dtype=int)
pattern_dict['CD19'] = np.asarray([0,0,0,0,0,1], dtype=int)

def parse_access_pattern(pattern):
    ''' The access pattern in Alvaro's data is a dash delimited string indicating in which cell-types this particular peak is accessible.
    I need to return this as an int8 nparray
    e.g if pattern is H1hesc-CD34-CD14-CD56-CD3-CD19, output is np.asarray([1,1,1,1,1,1])
        if pattern is CD34-CD14, output is np.asarray([0,1,1,0,0,0])
    
    '''
    arrays = tuple([pattern_dict[d] for d in pattern.split('-')])
    return np.sum(np.vstack(arrays), axis=0)
    
        
# produce Basset-style bed file for peaks, and activations text file for processing into a tensor-style data set





