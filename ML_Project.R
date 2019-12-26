rm(list = ls())
cat("\014")
set.seed(1)
library(MASS)
library(class)
library(caret)
library(glmnet)
library(randomForest)
library(gridExtra)
library(ggplot2)
library(e1071)
library(gridExtra)
library(ggpubr)
library(RCurl)


myfile_1 <- getURL("https://raw.githubusercontent.com/LingdiHuang/ML_Final/master/df_sample.csv")
df <- read.csv(textConnection(myfile_1),header=T)
df = df[,-1]
sample2 = sample(dim(df)[1], dim(df)[1]*.9)
dfTrain2 = df[sample2,]
dfTest2 = df[-sample2,]
XTrain2 = model.matrix(V120 ~ .,dfTrain2)[,-1]
XTest2 = model.matrix(V120 ~ .,dfTest2)[,-1]

sample1 = sample(dim(df)[1], dim(df)[1]*.5)
dfTrain1 = df[sample1,]
dfTest1 = df[-sample1,]
XTrain1 = model.matrix(V120 ~ .,dfTrain1)[,-1]
XTest1 = model.matrix(V120 ~ .,dfTest1)[,-1]

#Sample2 record time, cv curve and heat map
########################################################################
m = 25
ridge.cv2 = cv.glmnet(XTrain2, dfTrain2$V120, family = "binomial", alpha = 0,  intercept = TRUE, 
                      standardize = FALSE,  nfolds = 10, type.measure="class")
lam.ridge2 = exp(seq(log(max(ridge.cv2$lambda)),log(0.00001), -(log(max(ridge.cv2$lambda))-log(0.00001))/(m-1)))

##Record Time for cross validation
ptm = proc.time()
ridge.cv2 = cv.glmnet(XTrain2, dfTrain2$V120, lambda = lam.ridge2, family = "binomial", alpha = 0,  
                      intercept = TRUE, standardize = FALSE,  nfolds = 10, type.measure="class")
ptm = proc.time() - ptm
time.ridge.cv2 = ptm["elapsed"] 

##Record Time for fitting a single model
ptm = proc.time()
ridge.fit2 = glmnet(XTrain2, dfTrain2$V120, lambda = ridge.cv2$lambda, family = "binomial", 
                    alpha = 0, intercept = TRUE, standardize = FALSE)
ptm = proc.time() - ptm
time.ridge.fit2 = ptm["elapsed"] 

##For CV Curve
ridge.fit.0_2 = glmnet(XTrain2, dfTrain2$V120, lambda = 0, family = "binomial", alpha = 0,  
                       intercept = TRUE, standardize = FALSE)

n.lambdas2 = dim(ridge.fit2$beta)[2]
ridge.beta.ratio2 = rep(0, n.lambdas2)
for (i in 1:n.lambdas2) {
  ridge.beta.ratio2[i] = sqrt(sum((ridge.fit2$beta[,i])^2)/sum((ridge.fit.0_2$beta)^2))
}
##Lasso
lasso.cv2 = cv.glmnet(XTrain2, dfTrain2$V120, family = "binomial", alpha = 1,  intercept = TRUE, 
                      standardize = FALSE,  nfolds = 10, type.measure="class")
lam.lasso2 = exp(seq(log(max(lasso.cv2$lambda)),log(0.00001), -(log(max(lasso.cv2$lambda))-log(0.00001))/(m-1)))

##Record time for Cross Validation lasso
ptm = proc.time()
lasso.cv2 = cv.glmnet(XTrain2, dfTrain2$V120, lambda = lam.lasso2, family = "binomial", alpha = 1,  
                      intercept = TRUE, standardize = FALSE,  nfolds = 10, type.measure="class")
ptm = proc.time() - ptm
time.lasso.cv2 = ptm["elapsed"] 

##Record time for fitting a model
ptm = proc.time()
lasso.fit2 = glmnet(XTrain2, dfTrain2$V120, lambda = lasso.cv2$lambda, family = "binomial", 
                    alpha = 1, intercept = TRUE, standardize = FALSE)
ptm = proc.time() - ptm
time.lasso.fit2 = ptm["elapsed"] 

##Prepare for CV Curve
lasso.fit.0_2 = glmnet(XTrain2, dfTrain2$V120, lambda = 0, family = "binomial", alpha = 1,  
                       intercept = TRUE, standardize = FALSE)

n.lambdas2 = dim(lasso.fit2$beta)[2]
lasso.beta.ratio2 = rep(0, n.lambdas2)
for (i in 1:n.lambdas2) {
  lasso.beta.ratio2[i] = sqrt(sum((lasso.fit2$beta[,i])^2)/sum((lasso.fit.0_2$beta)^2))
}

###svm
##Record time for cv svm
ptm = proc.time()
svm.cv2 = tune(svm, V120~., data=dfTrain2, kernel = "radial",
               ranges = list(cost = 10^seq(-2,2,length.out = 5),
                             gamma = 10^seq(-2,2,length.out = 5)))
ptm = proc.time() - ptm
time.svm.cv2 = ptm["elapsed"] 

##Record time for fitting a svm
ptm = proc.time()
svm.fit2 = svm(V120~., data = dfTrain2, kernel = "radial", cost = svm.cv2$performances[,1], 
               gamma = svm.cv2$performances[,2])
ptm = proc.time() - ptm
time.svm.fit2 = ptm["elapsed"]

##Combine data for plot
eror2 = data.frame(c(rep("lasso", length(lasso.beta.ratio2)),  rep("ridge", length(ridge.beta.ratio2)) ), 
                   c(lasso.beta.ratio2, ridge.beta.ratio2) ,
                   c(lasso.cv2$cvm, ridge.cv2$cvm),
                   c(lasso.cv2$cvsd, ridge.cv2$cvsd))
colnames(eror2) = c("Method", "Ratio", "Cross-Validation Error", "sd")

##Plot
eror2.plot = ggplot(eror2, aes(x=Ratio, y = `Cross-Validation Error`, color=Method)) + geom_line(size=1)+ ggtitle('CV Curve For Lasso&Ridge (n_learn=0.9n)')+
  theme(plot.title = element_text(hjust=0.5))+
  theme(axis.title.x = element_text(size=16),
        axis.title.y = element_text(size=16),
        axis.text.x  = element_text(angle=0, vjust=0.5, size=12),
        axis.text.y  = element_text(angle=0, vjust=0.5, size=12))+
  geom_line(aes(linetype=Method), size=1)+
  geom_point(aes(shape=Method),size=4)

##Heat Map
SVM.df2 = svm.cv2$performances
SVM.df2$cost = as.character(SVM.df2$cost)
SVM.df2$gamma = as.character(SVM.df2$gamma)
SVM.plot2 = ggplot(SVM.df2, aes(x = gamma, y = cost, fill = error)) + geom_tile()+
  ggtitle(("SVM Heat Map"))+theme(plot.title = element_text(hjust=0.5))+
  theme(axis.title.x = element_text(size=16),
        axis.title.y = element_text(size=16),
        axis.text.x  = element_text(angle=0, vjust=0.5, size=12),
        axis.text.y  = element_text(angle=0, vjust=0.5, size=12))

##RF
train2 = dfTrain2
train2$V120 = as.factor(dfTrain2$V120)
ptm = proc.time()
rf.fit2 = randomForest(V120~., data = train2, mtry = sqrt(561))
ptm = proc.time() - ptm
time.tree.fit2 = ptm["elapsed"]

#Sample1 record time, cv curve and heat map
######################################################################## 
m = 25
ridge.cv1 = cv.glmnet(XTrain1, dfTrain1$V120, family = "binomial", alpha = 0,  intercept = TRUE, 
                      standardize = FALSE,  nfolds = 10, type.measure="class")
lam.ridge1 = exp(seq(log(max(ridge.cv1$lambda)),log(0.00001), -(log(max(ridge.cv1$lambda))-log(0.00001))/(m-1)))

##Record Time for cross validation
ptm = proc.time()
ridge.cv1 = cv.glmnet(XTrain1, dfTrain1$V120, lambda = lam.ridge1, family = "binomial", alpha = 0,  
                      intercept = TRUE, standardize = FALSE,  nfolds = 10, type.measure="class")
ptm = proc.time() - ptm
time.ridge.cv1 = ptm["elapsed"] 


##Record Time for fitting a single model
ptm = proc.time()
ridge.fit1 = glmnet(XTrain1, dfTrain1$V120, lambda = ridge.cv1$lambda, family = "binomial", 
                    alpha = 0, intercept = TRUE, standardize = FALSE)
ptm = proc.time() - ptm
time.ridge.fit1 = ptm["elapsed"] 


##For CV Curve
ridge.fit.0_1 = glmnet(XTrain1, dfTrain1$V120, lambda = 0, family = "binomial", alpha = 0,  
                       intercept = TRUE, standardize = FALSE)

n.lambdas1 = dim(ridge.fit1$beta)[2]
ridge.beta.ratio1 = rep(0, n.lambdas1)
for (i in 1:n.lambdas1) {
  ridge.beta.ratio1[i] = sqrt(sum((ridge.fit1$beta[,i])^2)/sum((ridge.fit.0_1$beta)^2))
}

lasso.cv1 = cv.glmnet(XTrain1, dfTrain1$V120, family = "binomial", alpha = 1,  intercept = TRUE, 
                      standardize = FALSE,  nfolds = 10, type.measure="class")
lam.lasso1 = exp(seq(log(max(lasso.cv1$lambda)),log(0.00001), -(log(max(lasso.cv1$lambda))-log(0.00001))/(m-1)))

ptm = proc.time()
lasso.cv1 = cv.glmnet(XTrain1, dfTrain1$V120, lambda = lam.lasso1, family = "binomial", alpha = 1,  
                      intercept = TRUE, standardize = FALSE,  nfolds = 10, type.measure="class")
ptm = proc.time() - ptm
time.lasso.cv1 = ptm["elapsed"] 

ptm = proc.time()
lasso.fit1 = glmnet(XTrain1, dfTrain1$V120, lambda = lasso.cv1$lambda, family = "binomial", 
                    alpha = 1, intercept = TRUE, standardize = FALSE)
ptm = proc.time() - ptm
time.lasso.fit1 = ptm["elapsed"] 

lasso.fit.0_1 = glmnet(XTrain1, dfTrain1$V120, lambda = 0, family = "binomial", alpha = 1,  
                       intercept = TRUE, standardize = FALSE)

n.lambdas1 = dim(lasso.fit1$beta)[2]
lasso.beta.ratio1 = rep(0, n.lambdas1)
for (i in 1:n.lambdas1) {
  lasso.beta.ratio1[i] = sqrt(sum((lasso.fit1$beta[,i])^2)/sum((lasso.fit.0_1$beta)^2))
}

error1 = data.frame(c(rep("lasso", length(lasso.beta.ratio1)),  rep("ridge", length(ridge.beta.ratio1)) ), 
                   c(lasso.beta.ratio1, ridge.beta.ratio1) ,
                   c(lasso.cv1$cvm, ridge.cv1$cvm),
                   c(lasso.cv1$cvsd, ridge.cv1$cvsd))
colnames(error1) = c("Method", "Ratio", "Cross-Validation Error Rate", "sd")

error1.plot = ggplot(error1, aes(x=Ratio, y = `Cross-Validation Error Rate`, color=Method)) + geom_line(size=1) + ggtitle('CV Curve For Lasso&Ridge (n_learn=0.5n)')+theme(plot.title = element_text(hjust=0.5))+
  theme(axis.title.x = element_text(size=16),
        axis.title.y = element_text(size=16),
        axis.text.x  = element_text(angle=0, vjust=0.5, size=12),
        axis.text.y  = element_text(angle=0, vjust=0.5, size=12))+
  geom_line(aes(linetype=Method), size=1)+
  geom_point(aes(shape=Method),size=4)

##Record time for tree
train = dfTrain1
train$V120 = as.factor(dfTrain1$V120)
ptm = proc.time()
rf.fit = randomForest(V120~., data = train, mtry = sqrt(561))
ptm = proc.time() - ptm
time.tree.fit = ptm["elapsed"]

###svm
##Record time for cv svm
ptm = proc.time()
svm.cv1 = tune(svm, V120~., data=dfTrain1, kernel = "radial",
               ranges = list(cost = 10^seq(-2,2,length.out = 5),
                             gamma = 10^seq(-2,2,length.out = 5)))
ptm = proc.time() - ptm
time.svm.cv1 = ptm["elapsed"] 

##Record time for fitting a svm
ptm = proc.time()
svm.fit1 = svm(V120~., data = dfTrain1, kernel = "radial", cost = svm.cv1$performances[,1], 
               gamma = svm.cv1$performances[,2])
ptm = proc.time() - ptm
time.svm.fit1 = ptm["elapsed"]

##Heat Map
SVM.df1 = svm.cv1$performances
SVM.df1$cost = as.character(SVM.df1$cost)
SVM.df1$gamma = as.character(SVM.df1$gamma)
SVM.plot1 = ggplot(SVM.df1, aes(x = gamma, y = cost, fill = error)) + geom_tile()+
  ggtitle(("SVM Heat Map (n_learn=0.5n)"))+theme(plot.title = element_text(hjust=0.5))+
  theme(axis.title.x = element_text(size=16),
        axis.title.y = element_text(size=16),
        axis.text.x  = element_text(angle=0, vjust=0.5, size=12),
        axis.text.y  = element_text(angle=0, vjust=0.5, size=12))


###############################Feature Importance#####################################
##Sample2
##Ridge features
tmp_coeffs_Ridge2 = coef(ridge.cv2, s = "lambda.min")
d_Ridge2 = data.frame(name = tmp_coeffs_Ridge2@Dimnames[[1]][tmp_coeffs_Ridge2@i + 1], 
                      coefficient = tmp_coeffs_Ridge2@x)[-1,]
d_Ridge2$coefficient = scale(d_Ridge2$coefficient)
d_Ridge2 = d_Ridge2[order(abs(d_Ridge2$coefficient), decreasing = TRUE),]
#d_Ridge2 = d_Ridge2[1:10, ]
d_Ridge2 = transform(d_Ridge2, name = reorder(name, abs(coefficient)))
d_Ridge2

##Lasso features
tmp_coeffs_Lasso2 = coef(lasso.cv2, s = "lambda.min") 
d_Lasso2 = data.frame(name = tmp_coeffs_Lasso2@Dimnames[[1]][tmp_coeffs_Lasso2@i + 1], 
                      coefficient = tmp_coeffs_Lasso2@x)[-1,]
d_Lasso2$coefficient = scale(d_Lasso2$coefficient)
d_Lasso2 = d_Lasso2[order(abs(d_Lasso2$coefficient), decreasing = TRUE),]
#d_Lasso2 = d_Lasso2[1:10, ]
d_Lasso2 = transform(d_Lasso2, name = reorder(name, abs(coefficient)))
d_Lasso2

##RF features
tmp_coeffs_RF2 = rf.fit2$importance
d_RF2 = data.frame(row.names(tmp_coeffs_RF2),tmp_coeffs_RF2)
d_RF2 = d_RF2[order(abs(d_RF2$MeanDecreaseGini), decreasing = TRUE),]
d_RF2 = d_RF2[1:10,]
colnames(d_RF2) = c("name", "importance")
d_RF2 = transform(d_RF2, name = reorder(name, abs(importance)))
d_RF2

##Sample1
##Ridge features
tmp_coeffs_Ridge1 = coef(ridge.cv1, s = "lambda.min")
d_Ridge1 = data.frame(name = tmp_coeffs_Ridge1@Dimnames[[1]][tmp_coeffs_Ridge1@i + 1], 
                      coefficient = tmp_coeffs_Ridge1@x)[-1,]
d_Ridge1$coefficient = scale(d_Ridge1$coefficient)
d_Ridge1 = d_Ridge1[order(abs(d_Ridge1$coefficient), decreasing = TRUE),]
#d_Ridge1 = d_Ridge1[1:10, ]
d_Ridge1 = transform(d_Ridge1, name = reorder(name, abs(coefficient)))
d_Ridge1

##Lasso Features
tmp_coeffs_Lasso1 = coef(lasso.cv1, s = "lambda.min") 
d_Lasso1 = data.frame(name = tmp_coeffs_Lasso1@Dimnames[[1]][tmp_coeffs_Lasso1@i + 1], 
                      coefficient = tmp_coeffs_Lasso1@x)[-1,]
d_Lasso1$coefficient = scale(d_Lasso1$coefficient)
d_Lasso1 = d_Lasso1[order(abs(d_Lasso1$coefficient), decreasing = TRUE),]
#d_Lasso1 = d_Lasso1[1:10, ]
d_Lasso1 = transform(d_Lasso1, name = reorder(name, abs(coefficient)))
d_Lasso1


##RF Features
tmp_coeffs_RF1 = rf.fit$importance
d_RF1 = data.frame(row.names(tmp_coeffs_RF1),tmp_coeffs_RF1)
d_RF1 = d_RF1[order(abs(d_RF1$MeanDecreaseGini), decreasing = TRUE),]
d_RF1 = d_RF1[1:10,]
colnames(d_RF1) = c("name", "importance")
d_RF1 = transform(d_RF1, name = reorder(name, abs(importance)))
d_RF1

#######Plot Feature importantce & Parameters############
df_coef_1 = rbind(d_Lasso1,d_Ridge1)
df_coef_1$Method = c(rep('Lasso',89),rep('Ridge',118))

##Sample 1
d_Lasso1$order = c(seq(1,89))
d_Ridge1$order = c(seq(1,118))

d_Lasso1_top10 = d_Lasso1[1:10,]
df_combine_features1 = merge(d_Lasso1_top10,d_Ridge1,by.x='name',by.y='name')
names(df_combine_features1) = c('Features','Coefficient_Lasso','order_Lasso','Coefficient_Ridge','order_Ridge')

g.mid<-ggplot(df_combine_features1,aes(x=1,y=Features))+geom_text(aes(label=Features),size=5)+
  geom_segment(aes(x=0.94,xend=0.96,yend=Features))+
  geom_segment(aes(x=1.04,xend=1.065,yend=Features))+
  ggtitle("")+
  ylab(NULL)+
  scale_x_continuous(expand=c(0,0),limits=c(0.94,1.065))+
  theme(axis.title=element_blank(),
        panel.grid=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.background=element_blank(),
        axis.text.x=element_text(color=NA),
        axis.ticks.x=element_line(color=NA),
        plot.margin = unit(c(1,-1,1,-1), "mm"))


g1 <- ggplot(data = df_combine_features1, aes(x = Features, y = Coefficient_Lasso)) +
  geom_bar(stat = "identity",width = 0.5) + ggtitle("Lasso(n_learn=0.5)") +
  theme(axis.title.x = element_blank(), 
        axis.title.y = element_blank(), 
        axis.text.y = element_blank(), 
        axis.ticks.y = element_blank(), 
        plot.margin = unit(c(1,-1,1,0), "mm")) +
  theme(plot.title = element_text(hjust=0.5,size=10,face='bold'))+
  scale_y_reverse() + coord_flip()

g2 <- ggplot(data = df_combine_features1, aes(x = Features, y = Coefficient_Ridge)) +xlab(NULL)+
  geom_bar(stat = "identity",width = 0.5) + ggtitle("Ridge(n_learn=0.5)") +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank(), 
        axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        plot.margin = unit(c(1,0,1,-1), "mm")) +
  theme(plot.title = element_text(hjust=0.5,size=10,face='bold'))+
  coord_flip()

gg1 <- ggplot_gtable(ggplot_build(g1))
gg2 <- ggplot_gtable(ggplot_build(g2))
gg.mid <- ggplot_gtable(ggplot_build(g.mid))

title1=text_grob("Coefficients Comparison (n_learn=0.5n)", face ="bold", size = 18)
grid.arrange(gg1,gg.mid,gg2,ncol=3,widths=c(4/9,1/9,4/9),top = title1)


##Sample 2
d_Lasso2$order = c(seq(1,68))
d_Ridge2$order = c(seq(1,118))

d_Lasso2_top10 = d_Lasso1[1:10,]
df_combine_features2 = merge(d_Lasso2_top10,d_Ridge1,by.x='name',by.y='name')
names(df_combine_features2) = c('Features','Coefficient_Lasso','order_Lasso','Coefficient_Ridge','order_Ridge')

g.mid2<-ggplot(df_combine_features2,aes(x=1,y=Features))+geom_text(aes(label=Features),size=5)+
  geom_segment(aes(x=0.94,xend=0.96,yend=Features))+
  geom_segment(aes(x=1.04,xend=1.065,yend=Features))+
  ggtitle("")+
  ylab(NULL)+
  scale_x_continuous(expand=c(0,0),limits=c(0.94,1.065))+
  theme(axis.title=element_blank(),
        panel.grid=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.background=element_blank(),
        axis.text.x=element_text(color=NA),
        axis.ticks.x=element_line(color=NA),
        plot.margin = unit(c(1,-1,1,-1), "mm"))


g1_2 <- ggplot(data = df_combine_features2, aes(x = Features, y = Coefficient_Lasso)) +
  geom_bar(stat = "identity",width = 0.5) + ggtitle("Lasso(n_learn=0.9n)") +
  theme(axis.title.x = element_blank(), 
        axis.title.y = element_blank(), 
        axis.text.y = element_blank(), 
        axis.ticks.y = element_blank(), 
        plot.margin = unit(c(1,-1,1,0), "mm")) +
  theme(plot.title = element_text(hjust=0.5,size=10,face='bold'))+
  scale_y_reverse() + coord_flip()

g2_2 <- ggplot(data = df_combine_features2, aes(x = Features, y = Coefficient_Ridge)) +xlab(NULL)+
  geom_bar(stat = "identity",width = 0.5) + ggtitle("Ridge(n_learn=0.9n)") +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank(), 
        axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        plot.margin = unit(c(1,0,1,-1), "mm")) +
  theme(plot.title = element_text(hjust=0.5,size=10,face='bold'))+
  coord_flip()

gg1_2 <- ggplot_gtable(ggplot_build(g1_2))
gg2_2 <- ggplot_gtable(ggplot_build(g2_2))
gg.mid2 <- ggplot_gtable(ggplot_build(g.mid2))

title2=text_grob("Coefficients Comparison (n_learn=0.9n)", face ="bold", size = 18)
grid.arrange(gg1_2,gg.mid2,gg2_2,ncol=3,widths=c(4/9,1/9,4/9),top=title2)

##RF
g.mid_rf<-ggplot(df_rf_combine,aes(x=1,y=name))+geom_text(aes(label=name),size=5)+
  geom_segment(aes(x=0.94,xend=0.96,yend=name))+
  geom_segment(aes(x=1.04,xend=1.065,yend=name))+
  ggtitle("")+
  ylab(NULL)+
  scale_x_continuous(expand=c(0,0),limits=c(0.94,1.065))+
  theme(axis.title=element_blank(),
        panel.grid=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.background=element_blank(),
        axis.text.x=element_text(color=NA),
        axis.ticks.x=element_line(color=NA),
        plot.margin = unit(c(1,-1,1,-1), "mm"))

g_rf_1 <- ggplot(data = d_RF1, aes(x = name, y = importance)) +
  geom_bar(stat = "identity",width = 0.5) + ggtitle("n_learn=0.5n") +
  theme(axis.title.x = element_blank(), 
        axis.title.y = element_blank(), 
        axis.text.y = element_blank(), 
        axis.ticks.y = element_blank(), 
        plot.margin = unit(c(1,-1,1,0), "mm")) +
  theme(plot.title = element_text(hjust=0.5,size=10,face='bold'))+
  scale_y_reverse() + coord_flip()

g_rf_2<-ggplot(data = d_RF2, aes(x = name, y = importance)) +xlab(NULL)+
  geom_bar(stat = "identity",width = 0.5) + ggtitle("n_learn=0.9n") +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank(), 
        axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        plot.margin = unit(c(1,0,1,-1), "mm")) +
  theme(plot.title = element_text(hjust=0.5,size=10,face='bold'))+
  coord_flip()

title3=text_grob("Feature Importance", face ="bold", size = 18)
grid.arrange(g_rf_1,g.mid_rf,g_rf_2,ncol=3,widths=c(4/9,1/9,4/9),top=title3)




  