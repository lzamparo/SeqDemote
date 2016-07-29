from optparse import OptionParser
import os
import sys


################################################################################
# generate_activations.py
#
# Create a tsv activations file for each of the flanks found in the flanks 
# fasta file.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <flanks_fa_file>'
    parser = OptionParser(usage)
    parser.add_option('-o', dest='act_output', default='encode_roadmap_flanks_act.txt', help='output file name. [Default: %default]')
    parser.add_option('-l', dest='output_dim', default=164, help='output dimension of the activations for each peak [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide fasta file providing BED-style sequences file paths.')
    else:
        flanks_fa_file = args[0]

    
    #################################################################
    # read in flanks bedfile, generate & write output to act file
    #################################################################   
    outfile = open(options.act_output,'w')
    
    with open(flanks_fa_file,'r') as f:
        for flank in f:
            if not flank.startswith('>'):
                continue
            flank_name = flank.lstrip('>').rstrip()
            flank_act_output = flank_name + '\t' + make_flank_acts(options.output_dim)
            print(flank_act_output, file=outfile)
            
    outfile.close()

def make_flank_acts(length):
    return '\t'.join(['0' for i in range(length)])

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
