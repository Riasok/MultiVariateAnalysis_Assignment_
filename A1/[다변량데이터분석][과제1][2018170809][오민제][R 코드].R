#
# NBA player salary estimation
# 
# MDA Assignment 1, MLR
# 2018170809 오민제

install.packages("dplyr")
install.packages("moments")
install.packages("ggplot2")
install.packages("corrplot")

library("dplyr")
library("moments")
library("ggplot2")
library("corrplot")

df <- read.csv("nba_contracts_history.csv") #dataset

nrow(unique(df["NAME"])) #number of players = 138
df$W <- df$W/df$GP; df$L <- NULL; names(df)[names(df) == "W"] <- "WR" #Winrate

for (i in c("MIN", "PTS", "FGM", "FGA", "X3PM", "X3PA", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF","MARGIN" )){
  df[i] <- df[i] / df["GP"]
}  ##divide the variables by game no.


stats <- data.frame(matrix(nrow=23, ncol=4)); names(stats) <- c("mean", "sd","kurtosis", "skewness")  #calculating the basic statistics
for (i in 1:23){
  stats[i,1] <- mean(unlist(df[names(df)[5:27][i]]))
  stats[i,2] <- sd(unlist(df[names(df)[5:27][i]]))
  stats[i,3] <- kurtosis(unlist(df[names(df)[5:27][i]]))
  stats[i,4] <- skewness(unlist(df[names(df)[5:27][i]]))
}
write.csv(stats, file = "stats.csv") #save
   

boxplot(df[names(df)[5:27][i]], main = names(df)[5:27][i])  #boxplots
pdf(file = "boxplots.pdf") #save 

#salary cap
salary <- c(58.040, 58.044, 58.044, 63.065, 70.000, 94.143, 99.093, 101.869, 109.140, 109.140)
years <- c("11-12","12-13","13-14","14-15","15-16","16-17", "17-18", "18-19", "19-20", "20-21")
bp <- barplot(salary, horiz = T,main="NBA Salary Cap", ylab = "season", xlab="million $",names.arg = years) 
text(x=bp, y=salary+5,labels = as.character(salary), col = "black", cex=1)

##salary adjustment
salary.cap <- data.frame("years" = c(2011,2012,2013,2014,2015,2016,2017,2018,2019,2020), "cap" = c(58.04,58.044, 58.044, 63.065, 70, 94.143, 99.093, 101.869, 109.14, 109.14))
df$Salary.ratio <- c(0)
for (i in 1:length(df$NAME)){
  df$Salary.ratio[i] <- df$AVG_SALARY[i] / (salary.cap[which(salary.cap[1] == df$CONTRACT_START[i]),2]*100)
}

#normality test
shapiro.result <- data.frame(matrix(nrow=23, ncol=2)); names(shapiro.result) <- c("W", "p-value") #shapiro- wilks
for(i in 1:23){
  shapiro.result[i,1] <- shapiro.test(unlist(df[names(df)[5:27][i]]))[1]
  shapiro.result[i,2] <- shapiro.test(unlist(df[names(df)[5:27][i]]))[2]
}
write.csv(shapiro.result, file = "shapiro.csv") #save
qqnorm(unlist(df[names(df)[5:27][i]]), main = names(df)[5:27][i]); qqline(unlist(df[names(df)[5:27][i]])) #QQPLOT

detect_outlier <- function(x) { ##Anomaly detection - from barplot Q3+1.5IQR, Q1-1.5*IQR
  Quantile1 <- quantile(x, probs=.25)
  Quantile3 <- quantile(x, probs=.75)
  IQR = Quantile3-Quantile1
  return(c(Quantile3 + 1.5*IQR, Quantile1 - 1.5*IQR))
}


outliers <- NULL  ##creating a dataframe with ouliers, where anything over Q3 +1.5IQR = 1, under = -1, rest = 0
for (i in 1:23){
  temp <- df %>% filter(df[names(df)[5:27][i]] > detect_outlier(unlist(df[names(df)[5:27][i]]))[1]);
  if (nrow(temp) != 0){
    temp[5:27] <- c(0)
    temp[i+4] <- c(1)
    outliers <- rbind(outliers,temp)
  }
  temp <- df %>% filter(df[names(df)[5:27][i]] < detect_outlier(unlist(df[names(df)[5:27][i]]))[2])
  if (nrow(temp != 0)){
    temp[5:27] <- c(0)
    temp[i+4] <- c(-1)
    outliers <- rbind(outliers, temp)
  }
}
outliers <- outliers %>%group_by(NAME, CONTRACT_START) %>% summarise(sum(AGE), sum(GP), sum(WR), sum(MIN), sum(PTS), sum(FGM), sum(FGA), sum(FG.), sum(X3PM), sum(X3PA), sum(X3P.), sum(FTM),sum(FTA), sum(FT.), sum(OREB), sum(DREB), sum(REB), sum(AST), sum(TOV), sum(STL), sum(BLK), sum(PF), sum(MARGIN))
outliers$sum <- c(0)
for (i in 1:79){
  outliers$sum[i] <- length(which(outliers[i,3:25] != 0))
}
outliers <- outliers %>% filter(sum > 2)
df <- left_join(df, outliers[c("NAME","CONTRACT_START", "sum")])
df[-which(!(is.na(df$sum))),] -> df

panel.points<-function(x,y){
  points(x,y,cex=0.05)
}
x<-c(0,1,2,3,4,5,6)
y<-c(120,167,188,190,192,198,199)
plot(x=x, y=y, type="b")


#pair-plot
pairs(df[3:27],lower.panel=panel.points,upper.panel = NULL , gap = 0.25)

#divide dataset
set.seed(1)
df.var <- df[5:28]
train.idx <- sample(1:186, round(0.7*186))
df.train <- df.var[train.idx,]
df.val <- df.var[-train.idx,]
#MLR
mlr.df <- lm(Salary.ratio ~., data = df.train)
summary(mlr.df)
plot(mlr.df)
plot(df.train$Salary.ratio, fitted(mlr.df), xlab="salary ratio", ylab="fitted")
abline(0,1, lty=3)

mlr.res <- resid(mlr.df)
hist(mlr.res, breaks = 50, prob=TRUE, xlab="x-variable")
curve(dnorm(x, mean=mean(mlr.res), sd=sd(mlr.res)), col="darkblue", lwd=2, add=TRUE, yaxt='n')
      
      
      
#variable reduction
df2 <- df.var[c(3,6,12,17,18,24)]
train.idx2 <- sample(1:186, round(0.7*186))
df2.train <- df2[train.idx2,]
df2.val <- df2[-train.idx2,]
mlr.df2 <- lm(Salary.ratio ~., data = df2.train)
summary(mlr.df2)
plot(mlr.df2)
plot(df2.train$Salary.ratio, fitted(mlr.df2), xlab="salary ratio", ylab="fitted")
abline(0,1, lty=3)

mlr.res2 <- resid(mlr.df2)
hist(mlr.res2, breaks = 50, prob=TRUE, xlab="x-variable")
curve(dnorm(x, mean=mean(mlr.res2), sd=sd(mlr.res2)), col="darkblue", lwd=2, add=TRUE, yaxt='n')

#perfomance
per.eval <- function(tgt.y, pre.y){
  #RMSE
  rmse <- sqrt(mean(tgt.y - pre.y)^2)
  #MAE
  mae <- mean(abs((tgt.y - pre.y)))
  #MAPE
  mape <- 100*mean(abs((tgt.y - pre.y)/tgt.y))
  return(c(rmse, mae, mape))
}
perf.mat <- matrix(0, nrow=2, ncol=3)
rownames(perf.mat) <- c("Full", "Reduced")
colnames(perf.mat) <- c("RMSE", "MAE", "MAPE")

mlr.pred1 <- predict(mlr.df, newdata = df.val)
mlr.pred2 <- predict(mlr.df2, newdata = df2.val)

perf.mat[1,] <- per.eval(df.val$Salary.ratio, mlr.pred1)
perf.mat[2,] <- per.eval(df2.val$Salary.ratio, mlr.pred2)

x <- 0; y<-0
#repetitive perfo check
for (i in 1:10){
  train.idx <- sample(1:186, round(0.7*186))
  df.train <- df.var[train.idx,]
  df.val <- df.var[-train.idx,]
  mlr.df <- lm(Salary.ratio ~., data = df.train)
  
  train.idx2 <- sample(1:186, round(0.7*186))
  df2.train <- df2[train.idx2,]
  df2.val <- df2[-train.idx2,]
  mlr.df2 <- lm(Salary.ratio ~., data = df2.train)
  
  mlr.pred1 <- predict(mlr.df, newdata = df.val)
  mlr.pred2 <- predict(mlr.df2, newdata = df2.val)
  
  perf.mat[1,] <- per.eval(df.val$Salary.ratio, mlr.pred1) + perf.mat[1,]
  perf.mat[2,] <- per.eval(df2.val$Salary.ratio, mlr.pred2) + perf.mat[2,]
  x <- x + as.numeric(summary(mlr.df)[8])
  y <- y+ as.numeric(summary(mlr.df2)[8])
}

perf.mat[1,] <- perf.mat[1,]/10
perf.mat[2,] <- perf.mat[2,]/10


