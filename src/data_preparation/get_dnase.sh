#!/bin/sh

# ENCODE
wget -r ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgDnaseUniform

# rearrange
mv hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgDnaseUniform encode
rm -r hgdownload.cse.ucsc.edu

# make sample-BED table
./make_encode_beds.py

# Roadmap
wget -r -A "*DNase.hotspot.fdr0.01.peaks.bed.gz" http://egg2.wustl.edu/roadmap/data/byFileType/peaks/consolidated/narrowPeak

# rearrange
mv egg2.wustl.edu/roadmap/data/byFileType/peaks/consolidated/narrowPeak roadmap
rm -r egg2.wustl.edu
rmdir roadmap/hammock

# make sample-BED table
./make_roadmap_beds.py

# combine
cat encode_beds.txt roadmap_beds.txt > sample_beds.txt
