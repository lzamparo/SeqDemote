library(data.table)
library(ggplot2)

setwd("/Users/zamparol/projects/SeqDemote/data/ATAC/mouse_asa/mapped_reads/CD8_effector")
peak_stats_dt = data.table(fread("peak_stats.csv"))

# select a random 2k peaks
peak_set = sample(peak_stats_dt[,peak], size = 500, replace=FALSE)
peak_stats_reduced = peak_stats_dt[peak %in% peak_set,]
ggplot(peak_stats_reduced, aes(x=peak, y=difference)) + geom_boxplot()
