require(cqn)

### apply quantile normalization -> conditional quantile normalization (see Corces et al.) to the count table for their atlas

# read in the count table, gc content info
setwd('/Users/zamparol/projects/SeqDemote/data/ATAC')
gc_bed <- read.csv(file = "GSE74912_ATACseq_peaks_GCcontent.bed", sep="\t")
colnames(gc_bed) <- c("chr", "start", "end", "pct_at", "pcg_gc", "num_A", "num_C", "num_G", "num_T", "num_N", "num_oth", "length")
normal_count_table <- read.csv("GSE74912_ATACseq_Normal_Counts_Recoded.txt", sep='\t', check.names = FALSE)
just_counts <- normal_count_table[,4:83]

# quantile normalize the counts
qn_normal_count_table <- normalize.quantiles(as.matrix(just_counts))

# compute the CQN coefficients
cqn_normal_count_table <- cqn(qn_normal_count_table, gc_bed$pcg_gc, gc_bed$length, lengthMethod="fixed")

# get the CQN normalized counts
cqn_normalized_values <- cqn_normal_count_table$y + cqn_normal_count_table$offset
norm_values_df <- as.data.frame(cqn_normalized_values)
named_norm_values_df <- data.frame(cbind(normal_count_table[,1:3], norm_values_df))

# write them out
write.table(named_norm_values_df, file = "cqn_normalized_activations_labeled.tsv", quote=FALSE, sep = "\t", row.names=FALSE, col.names=TRUE)
