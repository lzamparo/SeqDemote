from optparse import OptionParser
import os
import sys
from intervaltree import Interval, IntervalTree

################################################################################
# preprocess_flanks.py
#
# Produce a set of true negatives flanks for Basset analysis, potentially adding
# them to an existing database of features, specified as a BED file with the
# target activities comma-separated in column 4 and a full activity table file.
################################################################################

################################################################################
# main
################################################################################
def make_flanks(my_args=None):
    usage = 'usage: %prog [options] <target_beds_file>'
    parser = OptionParser(usage)
    parser.add_option('-o', dest='flank_bed', default='encode_roadmap_flanks', help='prefix for output bed_file')
    parser.add_option('-l', dest='chrom_lengths', default='human.hg19.genome', help='human chromosome lengths file')
    parser.add_option('-p', dest='chrom_path', default='/Users/zamparol/projects/SeqDemote/data/genomes', help='path to chromosome lengths file')
    parser.add_option('-m', dest='merge_overlap', default=200, type='int', help='Overlap length (after extension to feature_size) above which to merge features [Default: %default]')
    parser.add_option('-s', dest='feature_size', default=600, type='int', help='Extend features to this size [Default: %default]')
    if not my_args:
        (options,args) = parser.parse_args()
    else:
        (options, args) = parser.parse_args(args=my_args)
    if len(args) != 1:
        parser.error('Must provide file labeling the targets and providing BED file paths.')
    else:
        target_beds_file = args[0]

    chrom_trees = {}
    
    #################################################################
    # read in peaks bedfile, and build chromosome by chromosome IntervalTrees
    #################################################################        
    with open(target_beds_file,'r') as f:
        for peak in f:
            chrom, start, end, *rest = peak.split()
            chrom = chrom.strip()
            if chrom not in chrom_trees.keys():
                chrom_trees[chrom] = IntervalTree()
            chrom_trees[chrom][int(start):int(end)] = "%d-%d".format(start,end)
            

    #################################################################
    # read chromosome lengths
    #################################################################
    chrom_lengths = {}
    with open(os.path.join(options.chrom_path, options.chrom_lengths), 'r') as f:
        for line in f:
            chrom, length = line.strip().split()
            chrom_lengths[chrom] = int(length)
        

    #################################################################
    # build a list of flank regions that do not overlap any peak in any cell type
    #################################################################
    my_flanks = []
    with open(target_beds_file,'r') as f:
        for peak in f:
            chrom, start, end, *rest = peak.split()
            chrom = chrom.strip()
            up, down = flanks_from_peak(int(start), int(end), chrom, options.feature_size)
            up_overlaps = chrom_trees[chrom][up.start:up.end]
            down_overlaps = chrom_trees[chrom][down.start:down.end]
            if len(up_overlaps) == 0 and is_valid_flank(up, chrom_lengths):
                my_flanks.append(up)
            if len(down_overlaps) == 0 and is_valid_flank(down, chrom_lengths):
                my_flanks.append(down)
    
    #################################################################
    # print the flanks to a .bed file
    #################################################################    
    if not options.flank_bed.endswith(".bed"):
        outfile = options.flank_bed + '.bed'
    else:
        outfile = options.flank_bed
    with open(outfile, 'w') as f:
        for flank in my_flanks:
            print(flank, file=f)


def flanks_from_peak(start,end, chrom, length=600):
    upstream = Flank(chrom, start - length, start, set())
    downstream = Flank(chrom, end, end + length, set())
    return upstream, downstream

def is_valid_flank(f, chrom_lengths):
    return f.start > 0 and f.end < chrom_lengths[f.chrom]

class Flank:
    ''' flank representation

    Attributes:
        start (int) : flank start
        end   (int) : flank end
        chrom (str) : chromosome on which this peak is located
        act   (set[int]) : set of target indexes where this flank is active.  This should always be the empty set.
    '''
    def __init__(self, chrom, start, end, act):
        self.start = start
        self.end = end
        self.act = act
        self.chrom = chrom
        self.strand = '+'

    def extend(self, ext_len, chrom_len):
        ''' Extend the flank to the given length

        Args:
            ext_len (int) : length to extend the flank to
            chrom_len (int) : chromosome length to cap the flank at
        '''
        mid = (self.start + self.end - 1) / 2.0
        mid = int(0.5 + mid)
        self.start = max(0, mid - ext_len/2)
        self.end = self.start + ext_len
        if chrom_len and self.end > chrom_len:
            self.end = chrom_len
            self.start = self.end - ext_len

    def __str__(self):
        ''' Return a BED-style line'''
        return self.bed_str(self.chrom, self.strand)
        
    def bed_str(self, chrom, strand):
        ''' Return a BED-style line

        Args:
            chrom (str)
            strand (str)
        '''
        if len(self.act) == 0:
            act_str = '.'
        else:
            act_str = ','.join([str(ai) for ai in sorted(list(self.act))])
        cols = (chrom, str(self.start), str(self.end), '.', '1', strand, act_str)
        return '\t'.join(cols)

    def merge(self, flank2, ext_len, chrom_len):
        ''' Merge the given flank2 into this flank

        Args:
            flank2 (flank)
            ext_len (int) : length to extend the merged flank to
            chrom_len (int) : chromosome length to cap the flank at
        '''
        # find flank midpoints
        flank_mids = [(self.start + self.end - 1) / 2.0]
        flank_mids.append((flank2.start + flank2.end - 1) / 2.0)

        # weight flanks
        flank_weights = [1+len(self.act)]
        flank_weights.append(1+len(flank2.act))

        # compute a weighted average
        merge_mid = int(0.5+np.average(flank_mids, weights=flank_weights))

        # extend to the full size
        merge_start = max(0, merge_mid - ext_len/2)
        merge_end = merge_start + ext_len
        if chrom_len and merge_end > chrom_len:
            merge_end = chrom_len
            merge_start = merge_end - ext_len

        # merge activities
        merge_act = self.act | flank2.act

        # set merge to this flank
        self.start = merge_start
        self.end = merge_end
        self.act = merge_act


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    make_flanks()
