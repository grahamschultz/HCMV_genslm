library(Biostrings)
library(DescTools)

# Load data
seqs <- readDNAStringSet("data/genomesAligned.fasta")
scores <- read.csv("data/allScores.csv")

#Find high attention codons
codonList <- sapply(scores, function(i) {
  codons <- as.character(substr(seqs, i, i+2))
  codons <- codons[!grepl("-", codons)]
  sum(codons == codons[1]) / length(codons)
})

plot(x = scores$attention, y = codonList, ylab = "Codon Match Frequency", xlab = "Attention Score", xlim = c(1.5,3.5))
