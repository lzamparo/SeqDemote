#! /bin/bash


for bg in $(find . -name *_treat_pileup.bdg)
do
	outfile=$(echo $(basename $bg) | sed -e 's/pileup\.bdg/pileup_sorted.bdg/g')
	outdir=$(dirname $bg)	
	sort -k 1,1 -k 2,2n $bg > $outdir/$outfile
done
