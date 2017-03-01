#!/bin/bash

# get all the atac-seq data from the bomb paper

# craft a url using the SRR identifier to grab via wget
# e.g ftp://ftp-trace.ncbi.nih.gov/sra/sra-instant/reads/ByRun/sra/{SRR|ERR|DRR}/<first 6 characters of accession>/<accession>/<accession>.sra
# ftp://ftp-trace.ncbi.nih.gov/sra/sra-instant/reads/ByRun/sra/SRR/SRR119/SRR1192353/SRR1192353.sra

prefix="ftp://ftp-trace.ncbi.nih.gov/sra/sra-instant/reads/ByRun/sra/SRR"

for line in $(cat srr_heme.txt)
do
	fsix=${line:0:6}
	url=$(echo $prefix/$fsix/$line/$line.sra)
	wget $url
done

