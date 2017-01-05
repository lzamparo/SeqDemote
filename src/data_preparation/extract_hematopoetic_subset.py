import os
import pandas
import numpy as np

### Grab all cells within the hematopoetic lineage out of the Roadmap data used for Alvaro's paper

os.chdir('/Users/zamparol/projects/SeqDemote/data/DNase')


# read both peaks data file, DNase counts file
peaks = pandas.read_csv("peaksTable.txt", sep="\t")
counts = pandas.read_csv("DNaseCnts.txt", sep="\t")

def parse_access_pattern(pattern):
    ''' The access pattern in Alvaro's data is a dash delimited string indicating in which cell-types this particular peak is accessible.
    I need to return this as an int8 nparray
    e.g if pattern is H1hesc-CD34-CD14-CD56-CD3-CD19, output is np.asarray([1,1,1,1,1,1])
        if pattern is CD34-CD14, output is np.asarray([0,1,1,0,0,0])
    
    '''
    parts = pattern.split('-')
    

# produce Basset-style bed file for peaks, and activations text file for processing into a tensor-style data set





