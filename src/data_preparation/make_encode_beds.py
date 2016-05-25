#!/usr/bin/env python
from optparse import OptionParser

################################################################################
# make_encode_beds.py
#
# Extract names for samples from the files.txt descriptions.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    #parser.add_option()
    (options,args) = parser.parse_args()

    beds_out = open('encode_beds.txt', 'w')

    for line in open('encode/files.txt'):
        a = line.split('\t')

        bedfile = 'encode/%s' % a[0]
        bedfile = bedfile.replace('Huh7.5', 'Huh75')

        attrs = a[1].split(';')
        for i in range(len(attrs)):
            at = attrs[i].strip()
            if at.startswith('cell='):
                cell = at[5:]
            elif at.startswith('treatment='):
                treat = at[10:]
                if treat.find('_') != -1:
                    treat = treat[:treat.find('_')]
                if treat != 'None':
                    cell += '_%s' % treat

        cell = cell.replace('_RO01746','')
        cell = cell.replace('Adult_','')
        cell = cell.replace('_Mobilized','')

        print >> beds_out, '%s\t%s' % (cell, bedfile)

    beds_out.close()

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
