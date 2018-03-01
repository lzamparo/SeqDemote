library(data.table)
library(GenomicRanges)
library(stringr)
library(plyr)

# Extract Han's encoded peak list of GM12878 ChIP peaks, make into flat files so 
# sequence info can be extracted & interface with embedder

setwd("/Users/zamparol/projects/SeqDemote/data/ChIP/GM12878")
data = readRDS("GM12878_2016_pks.rds")

factors_list = data.table(fread("factors_list.txt"))

# ChIP experiments that do not fail the ENCODE quality check for some serious reason
quality_list = data.table(fread("compliant_experiments.csv"))
quality_list[, factor_name := tstrsplit(Experiment.target,"-",keep=1)]
quality_list[,df_index := paste(factor_name, "GM12878", File.accession, sep="_")]

# factors I actually care about
sublist = quality_list[factor_name %in% c("YY1","TCF12","JUND","SMC3","STAT3","IKZF1","SMAD5","STAT1","RAD51","TCF7","IKZF2"),]
my_sublist = data[sublist$df_index]

# write out the list of peaks for each
mapply(function(x, y) write.csv(x, paste0(y,"_peaks.csv"), row.names=FALSE, quote=FALSE), my_sublist, sublist$df_index)

