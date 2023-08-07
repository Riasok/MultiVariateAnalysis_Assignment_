#install.packages("dplyr")
library(dplyr)

#Performance eval -------------------
perf.eval <- function(cm){
  
  #TPR
  TPR <- cm[2,2]/sum(cm[2,])
  #Precision
  PRE <- cm[2,2]/sum(cm[,2])
  #TNR
  TNR <- cm[1,1]/sum(cm[1,])
  #Accuracy
  ACC <- (cm[1,1]+cm[2,2])/sum(cm)
  #BCR
  BCR <- sqrt(TNR*TPR)
  #F1- measure
  F1 <- 2*TPR*PRE/(TPR+PRE)
  
  return(c(TPR, PRE, TNR, ACC, BCR, F1))
}


#Performance table ------------------
perf.table <- matrix(0, nrow = 5, ncol = 6)
rownames(perf.table) <- c("Full Tree","Post-Pruning", "Pre-Pruning", "Neural Network", "LogisticR")
colnames(perf.table) <- c("TPR", "Precision", "TNR", "Accuracy", "BCR", "F1-measure")

#Load data & Preprocessing -------------
HA.record <- read.csv("Heart Attack Data Set.csv")
set.seed(1)

#divide the dataset 6:4
idx <- c(1:303)
trn.idx <- sample(x=1:303, size= 182, replace = F)
tst.idx <- idx[!idx %in% trn.idx]

input.idx <- c(1:13)
target.idx <- 14

HA.record$target <- as.factor(HA.record$target)

#CART --------------------------------
#install.packages("tree")
library(tree)

CART.trn <- data.frame(HA.record[trn.idx,])
CART.tst <- data.frame(HA.record[tst.idx,])

#train tree
CART.post <- tree(target ~., CART.trn)
summary(CART.post)

#plot the tree
plot(CART.post)
text(CART.post, pretty = 1)

CART.prey <- predict(CART.post, CART.tst, type="class")
CART.cm <- table(CART.tst$target, CART.prey)

perf.table[1,] <- perf.eval(CART.cm)
#Post-Pruning 
CART.post.cv <- cv.tree(CART.post, FUN = prune.misclass, K=3)

#plot pruning results
plot(CART.post.cv$size, CART.post.cv$dev, type = "b")
CART.post.cv

#selecting post-pruned model
CART.post.pruned <- prune.misclass(CART.post, best = 5)
plot(CART.post.pruned)
text(CART.post.pruned, pretty = 1)

#prediction
CART.post.prey <- predict(CART.post.pruned, CART.tst, type="class")
CART.post.cm <- table(CART.tst$target, CART.post.prey)

perf.table[2,] <- perf.eval(CART.post.cm)

#Pre-Pruning CART--------------------
#install.packages("party")
library(party)

#AUROC
#install.packages("ROCR")
library(ROCR)
val.idx <- sample(trn.idx, size = 60, replace = F)
trn.idx <- trn.idx[!trn.idx %in% val.idx]


CART.trn <- HA.record[trn.idx,]
CART.val <- HA.record[val.idx,]
CART.tst <- HA.record[tst.idx,]

#single tree with evaluation
min.criterion = c(0.9,0.95, 0.99, 0.99)
min.split = c(5,10,30)
max.depth = c(0,3,4,5)

CART.pre.search.result =
  matrix(0,length(min.criterion)*length(min.split)*length(max.depth),11)
colnames(CART.pre.search.result) <- c("min.criterion", "min.split", "max.depth",
                                      "TPR", "Precision", "TNR", "ACC", "BCR", "F1", "AUROC", "N_leaves")


iter.cnt = 1
for (i in 1:length(min.criterion)) {
  for ( j in 1:length(min.split)) {
    for ( k in 1:length(max.depth)) {
      cat("CART Min criterion:", min.criterion[i], ", Min split:",
          min.split[j], ", Max depth:", max.depth[k], "\n")
      tmp.control = ctree_control(mincriterion = min.criterion[i],
                                  minsplit = min.split[j], maxdepth = max.depth[k])
      tmp.tree <- ctree(target ~ ., data = CART.trn, controls = tmp.control)
      tmp.tree.val.prediction <- predict(tmp.tree, newdata = CART.val)
      tmp.tree.val.response <- treeresponse(tmp.tree, newdata = CART.val)
      tmp.tree.val.prob <- 1-unlist(tmp.tree.val.response,
                                    use.names=F)[seq(1,nrow(CART.val)*2,2)]
      tmp.tree.val.rocr <- prediction(tmp.tree.val.prob, CART.val$target)
      
      # Confusion matrix for the validation dataset
      tmp.tree.val.cm <- table(CART.val$target, tmp.tree.val.prediction)
      
      CART.pre.search.result[iter.cnt,1] = min.criterion[i]
      CART.pre.search.result[iter.cnt,2] = min.split[j]
      CART.pre.search.result[iter.cnt,3] = max.depth[k]
      
      # Performances from the confusion matrix
      CART.pre.search.result[iter.cnt,4:9] = perf.eval(tmp.tree.val.cm)
      
      # AUROC
      CART.pre.search.result[iter.cnt,10] = unlist(performance(tmp.tree.val.rocr,
                                                               "auc")@y.values)
      
      # Number of leaf nodes
      CART.pre.search.result[iter_cnt,11] = length(nodes(tmp.tree,
                                                         unique(where(tmp.tree))))
      iter.cnt = iter.cnt + 1
    }
  }
}

# Find the best set of parameters
CART.pre.search.result <- CART.pre.search.result[order(CART.pre.search.result[,10],
                                                       decreasing = T),]
CART.pre.search.result
best.criterion <- CART.pre.search.result[1,1]
best.split <- CART.pre.search.result[1,2]
best.depth <- CART.pre.search.result[1,3]

#best tree
tree.control <- ctree_control(mincriterion = best.criterion, minsplit = best.split, maxdepth = best.depth)

#train & valid dataset for best tree
CART.trn <- rbind(CART.trn, CART.val)
CART.pre <- ctree(target ~., data = CART.trn, controls = tree.control)
CART.pre.prediction <- predict(CART.pre, newdata = CART.tst)
CART.pre.response <- treeresponse(CART.pre, newdata= CART.tst)

#performance of the best tree
CART.pre.cm <- table(CART.tst$target, CART.pre.prediction)
perf.table[3,] <- perf.eval(CART.pre.cm)
perf.table

#confusion matrix
CART.cm
CART.post.cm
CART.pre.cm

#ROC curve
CART.prob <- 1-unlist(CART.post.response,use.names=F)[seq(1,nrow(CART.tst)*2,2)]
CART.rocr <- prediction(CART.post.prob, CART.tst$target)
CART.perf <- performance(CART.post.rocr, "tpr","fpr")

CART.post.prob <- 1-unlist(CART.post.pruned.response,use.names=F)[seq(1,nrow(CART.tst)*2,2)]
CART.post.rocr <- prediction(CART.post.pruned.prob, CART.tst$target)
CART.post.perf <- performance(CART.post.pruned.rocr, "tpr","fpr")

CART.pre.prob <- 1-unlist(CART.pre.response,use.names=F)[seq(1,nrow(CART.tst)*2,2)]
CART.pre.rocr <- prediction(CART.pre.prob, CART.tst$target)
CART.pre.perf <- performance(CART.pre.rocr, "tpr","fpr")

plot(CART.pre.perf, col=5, lwd = 3)

#plot best tree
plot(CART.pre, type = "simple")


#Logistic Regression -------------------------
HA.scaled <- data.frame(scale(HA.record[,1:13]),"target" = HA.record[,target.idx])
LR.trn <- HF.scaled[trn.idx,]
LR.tst <- HF.scaled[tst.idx,]

LR <- glm(target ~ .,family = "binomial", LR.trn)
summary(LR)
LR.response <- predict(LR, type="response", newdata = LR.tst)
LR.predicted <- rep(0, length(LR.response))
LR.predicted[which(LR.response >= 0.5)] <- 1
LR.cm <- table(unlist(LR.tst[target.idx]), LR.predicted)
perf.table[5,] <- perf.eval(LR.cm)



#Neural Network -------------------
#install.packages("nnet")
library(nnet)
HA.scaled$target <- as.factor(HA.scaled$target)

#ANN training
ann.trn.input <- HA.scaled[trn.idx,input.idx]
ann.trn.target <- class.ind(HA.scaled[trn.idx, target.idx])


# Candidate hyperparameters
nH <- c(5,15,25)
Decay <- c(5e-10, 5e-4, 5e-1)
Maxit <- c(100, 500, 1000)

# 3-fold cross validation index
val.idx <- sample(c(1:3), dim(ann.trn.input)[1], replace = TRUE, prob = rep(0.33,3))
val.perf <- matrix(0, length(nH)*length(Decay)*length(Maxit)*3, 4)


count = 1
for (i in 1:length(nH)) {
  for (k in 1:length(Decay)){
    for (l in 1:length(Maxit)){
      eval.fold <- 
      cat("Training ANN: the number of hidden nodes:", nH[i], "\n")
      for (j in c(1:3)) {
        # Training with the data in (k-1) folds
        tmp.trn.input <- ann.trn.input[which(val.idx != j),]
        tmp.trn.target <- ann.trn.target[which(val.idx != j),]
        tmp.nnet <- nnet(tmp.trn.input, tmp.trn.target, size = nH[i],
                         decay = Decay[k], maxit = Maxit[l])
        
      
        # Evaluate the model with the remaining 1 fold
        tmp.val.input <- ann.trn.input[which(val.idx == j),]
        tmp.val.target <- ann.trn.target[which(val.idx == j),]
        
        eval.fold <- rbind(eval.fold, cbind(max.col(predict(tmp.nnet, tmp.val.input)), 
                                            max.col(tmp.val.target)))
      }
      # nH
      val.perf[count,1:3] <- c(nH[i], Decay[k], Maxit[l])
      
      # AUROC
      tmp.rocr <- prediction(eval.fold[,1], eval.fold[,2])
      val.perf[count,4] <- unlist(performance(tmp.rocr,"auc")@y.values)
      
      count <- count + 1
    }
  }
}; rm(count)

ordered.val.perf <- val.perf[order(val.perf[,4], decreasing = TRUE),]
colnames(ordered.val.perf) <- c("nH", "Decay","Maxit", "AUROC")
ordered.val.perf
# Find the best number of hidden node
best <- ordered.val.perf[1,1:3]

# Test the ANN
HA.nnet <- nnet(ann.trn.input, ann.trn.target, size = best[1],
                    decay = best[2], maxit = best[3])


#evaluate
nn.cm <- table(max.col(predict(HA.nnet, HF.scaled[tst.idx,input.idx]))-1, HF.scaled[tst.idx, target.idx])
perf.table[4,] <- perf.eval(nn.cm)









#Load data & Preprocessing 2 -------------
GC.record <- read.csv("gender_classification_v7.csv")

#divide the dataset 8:2
idx <- c(1:5001)
trn.idx <- sample(x=1:5001, size= 4001, replace = F)
tst.idx <- idx[!idx %in% trn.idx]

input.idx <- c(1:7)
target.idx <- 8

GC.record$gender[which(GC.record$gender == "Male")] <- 0
GC.record$gender[which(GC.record$gender == "Female")] <- 1
GC.record$gender <- as.factor(GC.record$gender)

#CART2 ----------
val.idx <- sample(trn.idx, size = 1000, replace = F)
trn.idx <- trn.idx[!trn.idx %in% val.idx]


CART.trn <- GC.record[trn.idx,]
CART.val <- GC.record[val.idx,]
CART.tst <- GC.record[tst.idx,]

#single tree with evaluation
min.criterion = c(0.9,0.95, 0.99, 0.99)
min.split = c(5,10,30,50)
max.depth = c(0,3,4,5,10, 20)

CART.pre.search.result =
  matrix(0,length(min.criterion)*length(min.split)*length(max.depth),11)
colnames(CART.pre.search.result) <- c("min.criterion", "min.split", "max.depth",
                                      "TPR", "Precision", "TNR", "ACC", "BCR", "F1", "AUROC", "N_leaves")


iter.cnt = 1
for (i in 1:length(min.criterion)) {
  for ( j in 1:length(min.split)) {
    for ( k in 1:length(max.depth)) {
      cat("CART Min criterion:", min.criterion[i], ", Min split:",
          min.split[j], ", Max depth:", max.depth[k], "\n")
      tmp.control = ctree_control(mincriterion = min.criterion[i],
                                  minsplit = min.split[j], maxdepth = max.depth[k])
      tmp.tree <- ctree(gender ~ ., data = CART.trn, controls = tmp.control)
      tmp.tree.val.prediction <- predict(tmp.tree, newdata = CART.val)
      tmp.tree.val.response <- treeresponse(tmp.tree, newdata = CART.val)
      tmp.tree.val.prob <- 1-unlist(tmp.tree.val.response,
                                    use.names=F)[seq(1,nrow(CART.val)*2,2)]
      tmp.tree.val.rocr <- prediction(tmp.tree.val.prob, CART.val$gender)
      
      # Confusion matrix for the validation dataset
      tmp.tree.val.cm <- table(CART.val$gender, tmp.tree.val.prediction)
      
      CART.pre.search.result[iter.cnt,1] = min.criterion[i]
      CART.pre.search.result[iter.cnt,2] = min.split[j]
      CART.pre.search.result[iter.cnt,3] = max.depth[k]
      
      # Performances from the confusion matrix
      CART.pre.search.result[iter.cnt,4:9] = perf.eval(tmp.tree.val.cm)
      
      # AUROC
      CART.pre.search.result[iter.cnt,10] = unlist(performance(tmp.tree.val.rocr,
                                                               "auc")@y.values)
      
      # Number of leaf nodes
      CART.pre.search.result[iter_cnt,11] = length(nodes(tmp.tree,
                                                         unique(where(tmp.tree))))
      iter.cnt = iter.cnt + 1
    }
  }
}

# Find the best set of parameters
CART.pre.search.result <- CART.pre.search.result[order(CART.pre.search.result[,10],
                                                       decreasing = T),]
CART.pre.search.result
best.criterion <- CART.pre.search.result[1,1]
best.split <- CART.pre.search.result[1,2]
best.depth <- CART.pre.search.result[1,3]

#best tree
tree.control <- ctree_control(mincriterion = best.criterion, minsplit = best.split, maxdepth = best.depth)

#train & valid dataset for best tree
CART.trn <- rbind(CART.trn, CART.val)
CART.pre <- ctree(gender ~., data = CART.trn, controls = tree.control)
CART.pre.prediction <- predict(CART.pre, newdata = CART.tst)
CART.pre.response <- treeresponse(CART.pre, newdata= CART.tst)

#performance of the best tree
CART.pre.cm <- table(CART.tst$gender, CART.pre.prediction)
perf.table[3,] <- perf.eval(CART.pre.cm)
perf.table

#confusion matrix
CART.pre.cm

#ROC curve
CART.prob <- 1-unlist(CART.post.response,use.names=F)[seq(1,nrow(CART.tst)*2,2)]
CART.rocr <- prediction(CART.post.prob, CART.tst$target)
CART.perf <- performance(CART.post.rocr, "tpr","fpr")

CART.post.prob <- 1-unlist(CART.post.pruned.response,use.names=F)[seq(1,nrow(CART.tst)*2,2)]
CART.post.rocr <- prediction(CART.post.pruned.prob, CART.tst$target)
CART.post.perf <- performance(CART.post.pruned.rocr, "tpr","fpr")

CART.pre.prob <- 1-unlist(CART.pre.response,use.names=F)[seq(1,nrow(CART.tst)*2,2)]
CART.pre.rocr <- prediction(CART.pre.prob, CART.tst$target)
CART.pre.perf <- performance(CART.pre.rocr, "tpr","fpr")

plot(CART.pre.perf, col=5, lwd = 3)

#plot best tree
plot(CART.pre, type= "simple")

#Logistic Regression 2 -------------------------
GC.scaled <- data.frame(scale(GC.record[,input.idx]),"gender" = GC.record[,target.idx])
LR.trn <- GC.scaled[c(trn.idx,val.idx),]
LR.tst <- GC.scaled[tst.idx,]

LR <- glm(gender ~ .,family = "binomial", LR.trn)
summary(LR)
LR.response <- predict(LR, type="response", newdata = LR.tst)
LR.predicted <- rep(0, length(LR.response))
LR.predicted[which(LR.response >= 0.5)] <- 1
LR.cm <- table(unlist(LR.tst[target.idx]), LR.predicted)
perf.table[5,] <- perf.eval(LR.cm)



#Neural Network 2 -------------------
#install.packages("nnet")

#ANN training
ann.trn.input <- GC.scaled[trn.idx,input.idx]
ann.trn.target <- class.ind(GC.scaled[trn.idx, target.idx])


# Candidate hyperparameters
nH <- c(5,15,25)
Decay <- c(5e-20, 5e-10, 5e-4, 5e-1)
Maxit <- c(100, 500, 1000)

# 3-fold cross validation index
val.idx <- sample(c(1:4), dim(ann.trn.input)[1], replace = TRUE, prob = rep(0.25,4))
val.perf <- matrix(0, length(nH)*length(Decay)*length(Maxit)*4, 4)


count = 1
for (i in 1:length(nH)) {
  for (k in 1:length(Decay)){
    for (l in 1:length(Maxit)){
      eval.fold <- 
      cat("Training ANN: the number of hidden nodes:", nH[i], "\n")
      for (j in c(1:3)) {
        # Training with the data in (k-1) folds
        tmp.trn.input <- ann.trn.input[which(val.idx != j),]
        tmp.trn.target <- ann.trn.target[which(val.idx != j),]
        tmp.nnet <- nnet(tmp.trn.input, tmp.trn.target, size = nH[i],
                         decay = Decay[k], maxit = Maxit[l])
        
      
        # Evaluate the model with the remaining 1 fold
        tmp.val.input <- ann.trn.input[which(val.idx == j),]
        tmp.val.target <- ann.trn.target[which(val.idx == j),]
        
        eval.fold <- rbind(eval.fold, cbind(max.col(predict(tmp.nnet, tmp.val.input)), 
                                            max.col(tmp.val.target)))
      }
      # nH
      val.perf[count,1:3] <- c(nH[i], Decay[k], Maxit[l])
      
      # AUROC
      tmp.rocr <- prediction(eval.fold[,1], eval.fold[,2])
      val.perf[count,4] <- unlist(performance(tmp.rocr,"auc")@y.values)
      
      count <- count + 1
    }
  }
}; rm(count)

ordered.val.perf <- val.perf[order(val.perf[,4], decreasing = TRUE),]
colnames(ordered.val.perf) <- c("nH", "Decay","Maxit", "AUROC")
ordered.val.perf
# Find the best number of hidden node
best <- ordered.val.perf[1,1:3]

# Test the ANN
GC.nnet <- nnet(ann.trn.input, ann.trn.target, size = best[1],
                    decay = best[2], maxit = best[3])


#evaluate
nn.cm <- table(max.col(predict(GC.nnet, GC.scaled[tst.idx,input.idx]))-1, GC.scaled[tst.idx, target.idx])
perf.table[4,] <- perf.eval(nn.cm)




save.image("data.rdata")

