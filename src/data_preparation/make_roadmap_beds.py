#!/usr/bin/env python
from optparse import OptionParser
import os
import pandas as pd

################################################################################
# make_roadmap_beds.py
#
# Extract names for samples from the spreadsheet.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    #parser.add_option()
    (options,args) = parser.parse_args()

    df = pd.read_excel('jul2013.roadmapData.qc.xlsx', sheetname='Consolidated_EpigenomeIDs_summa')

    beds_out = open('roadmap_beds.txt', 'w')

    for i in range(df.shape[0]):
	eid = df.iloc[i,1]

	peaks_bed = 'roadmap/%s-DNase.hotspot.fdr0.01.peaks.bed.gz' % eid
	if os.path.isfile(peaks_bed):
	    print >> beds_out, df.iloc[i,5], '\t', peaks_bed

    beds_out.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
