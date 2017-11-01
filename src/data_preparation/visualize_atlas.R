library(ggplot2)
library(gridExtra)
library(ggExtra)
library(ggridges)
library(scales)
library(AtlasAnnotater)

setwd('~/projects/SeqDemote/results/diagnostic_plots/ATAC/')
pdf(file = "atlas_diagnostic_plots.pdf", width = 15, height = 13)

pltList <- list()

# Peaks by annotation
total = just_rel[,.N]
grouped_peaks = just_rel[,.N, by=annotation]
grouped_peaks[, percentage := N / total]
pba <- ggplot(grouped_peaks, aes(x=annotation, fill=annotation)) + 
  geom_col(aes(y=N)) + 
  geom_text(aes(x=annotation, y=N, label=percent(percentage)), vjust = -0.75) +
  scale_y_continuous(breaks=c(0,10000,25000,50000,75000,100000,125000,150000)) +
  ggtitle("Number of peaks by annotation", subtitle="(topped by percentage of total)") +
  ylab("Number of peaks") + 
  xlab("Peak annotation")

pltList[[1]] <- pba

# Peak length by chromosome
plc <- ggplot(just_rel[width < 1600,], aes(x = width, y = seqnames)) +
  geom_density_ridges(stat = "binline",bins=50) + 
  xlim(c(0,1800)) + 
  xlab("Peak length (bp)") + ylab("Chromosome") + 
  theme_ridges(grid = FALSE)

pltList[[2]] <- plc

# Fine-grained histogram of peak lengths
fgpl <- ggplot(just_rel[width < 1800,], aes(x = width, y = annotation)) +
  geom_density_ridges(aes(fill=annotation), stat = "binline",bins=80) +
  labs(title="Peak lengths by annotation")

pltList[[3]] <- fgpl

# Gene complexity plot: number of peaks / gene

# Count peaks / gene
gene_count = just_rel[annotation != "intergenic", .N, by = nearest_gene]
setnames(gene_count, "N", "count")

# Get gene length data
annot = loadAnnot("hg19")
tx_lens = GenomicFeatures::transcriptLengths(annot$txdb, with.cds_len = TRUE)
tx_lens_dt = data.table(tx_lens)
gene_lens = merge(tx_lens_dt, annot$genenames, by.x=c("gene_id"), by.y=c("gene_id"))
gene_lens[,avg_coding_len := mean(cds_len), by=gene_name]
gene_lens[, total_exons := max(nexon), by=gene_name]
gene_lens[, max_len := max(tx_len), by=gene_name]
gene_lens = gene_lens[!duplicated(gene_id),.(gene_id,gene_name,avg_coding_len,total_exons,max_len)]
gene_length_dt = merge(gene_count, merged_lens, by.x=c("nearest_gene"), by.y=c("gene_name"))
gene_length_dt[avg_coding_len > 0, coding := "coding"]
gene_length_dt[avg_coding_len == 0, coding := "non-coding"]
gene_length_dt[,peak_per_bp := count / max_len]

gcp = ggplot(gene_length_dt, aes(x=max_len,y=count)) +
  geom_point(aes(alpha=0.25)) + guides(alpha=FALSE) + 
  scale_x_log10() +
  ggtitle("Peaks per gene", subtitle="by gene length") + 
  xlab("Gene length (log10 bp)") + 
  ylab("Number of peaks") +
  facet_wrap(~coding)

# display the plots in a grid
grid.arrange(grobs=pltList, ncol=2)
dev.off()
