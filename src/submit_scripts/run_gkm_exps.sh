#! /bin/bash

#-k $kernel -d $mismatch -e $expname

#bsub -env "all, kernel=4,mismatch=2,expname=wgkm" < run_gkm.lsf
bsub -env "all, kernel=2,mismatch=2,expname=gkm" < run_gkm.lsf
bsub -env "all, kernel=2,mismatch=0,expname=gkm_d0" < run_gkm.lsf
bsub -env "all, kernel=4,mismatch=0,expname=wgkm_d0" < run_gkm.lsf
