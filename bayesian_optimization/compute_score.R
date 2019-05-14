# compute BIC score of a Bayesnet
# input: adjacency matrix, dataset
# output: BIC score

library("bnlearn", quietly = TRUE, warn.conflicts = FALSE)
data("asia")
args = commandArgs(trailingOnly = TRUE)
# print(args[1])
input.file = args[1]
output.file = args[2]
ad.mat = as.matrix(read.table(input.file, sep = " ", col.names = names(asia), row.names = names(asia)))
# print(ad.mat)
# print(dim(ad.mat))
net = empty.graph(names(asia))
amat(net) <- ad.mat
# print(net)
s = score(net, asia)
write(s, file = output.file)