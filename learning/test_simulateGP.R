library(MASS)
n <- 500
G <- rbind(c(0,1),    # 2 VARIABLE DAG
           c(0,0))    #
G <- rbind(c(0,1,1),  # 3 VARIABLE DAG
           c(0,0,0),  #
           c(0,1,0))  #

x <- sampleDataFromG(n, G)
plot(x)
pairs(x, cex = 0.5)
