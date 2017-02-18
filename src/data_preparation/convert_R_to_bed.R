### Extract data as bed file (one each for peaks, flanks), and activity table (also one for peaks, flanks)
require(Biostrings)
require(dplyr)

# bed file example
# chr1	1208992	1209592	10	1	+	0,1,2,3,4,5
# chr1	11120062	11120662	100	1	+	0,1,2,3,4,5
# chr1	161067622	161068222	1000	1	+	0,1,2,3,4,5
# chr4	84376575	84377175	10000	1	+	0,1,2,3,4,5
# chr12	66584446	66585046	100000	1	+	1,2
# chr12	67244774	67245374	100002	1	+	1
# chr12	67318760	67319360	100003	1	+	1

# activity table example
# peakID	H1hesc	CD34	CD14	CD56	CD3	CD19
# chr1:1208992-1209592(+)	1	1	1	1	1	1
# chr1:11120062-11120662(+)	1	1	1	1	1	1
# chr1:161067622-161068222(+)	1	1	1	1	1	1
# chr4:84376575-84377175(+)	1	1	1	1	1	1

setwd('/Users/zamparol/projects/SeqDemote/results/SeqGL')
top_dir <- getwd()
cell_order <- data.frame(names=c("H1hesc","CD34","CD14","CD56","CD3","CD19"),position=c(1,2,3,4,5,6))

rdata_to_text <- function(directory, position) {

  # load the data
  setwd(paste(top_dir,directory,sep='/'))
  load("train_test_data.Rdata")
  
  peak_rows <- train.test.data$train.inds$pos
  flank_rows <- train.test.data$train.inds$neg
  all_data <- data.frame(seqnames=seqnames(train.test.data$train.regions),
                         starts=start(ranges(train.test.data$train.regions))-1,
                         ends=end(ranges(train.test.data$train.regions)),
                         names=c(rep(".", length(train.test.data$train.regions))),
                         scores=c(rep(".", length(train.test.data$train.regions))),
                         strands=strand(train.test.data$train.regions))
  
  # make the bed file, activation file for peaks
  peaks_bed <- all_data[peak_rows,]
  write.table(peaks_bed, file = paste0(directory, "_peaks.bed"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
  peaks_activation_table <- peaks_bed %>% mutate(peakID = paste0(seqnames,":",starts,"-",ends,"(+)"), H1hesc = 0, CD34 = 0, CD14 = 0, CD56 = 0, CD3 = 0, CD19 = 0) %>% select(peakID, H1hesc, CD34,CD14,CD56,CD3,CD19) 
  peaks_activation_table[,position + 1] = 1   # +1 is for the offset due to peakID
  write.table(peaks_activation_table, file = paste0(directory, "_peaks_act.txt"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
  
  # make the bed file, activation file for flanks
  flanks_bed <- all_data[flank_rows,]
  write.table(flanks_bed, file = paste0(directory, "_flanks.bed"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
  flanks_activation_table <- flanks_bed %>% mutate(flankID = paste0(seqnames,":",starts,"-",ends,"(+)"), H1hesc = 0, CD34 = 0, CD14 = 0, CD56 = 0, CD3 = 0, CD19 = 0) %>% select(flankID, H1hesc, CD34,CD14,CD56,CD3,CD19) 
  write.table(flanks_activation_table, file = paste0(directory, "_flanks_act.txt"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
}

mapply(rdata_to_text, cell_order$names, cell_order$position)


