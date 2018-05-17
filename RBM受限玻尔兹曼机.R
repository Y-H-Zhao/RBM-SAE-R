setwd("D:/Neural Networks")
#RBM（Restricted Boltzmann Machine）是一种无监督学习模型，近似于
#拟合样本的概率密度函数。
#过去10年里，RBM已经广泛应用于跟多的应用中，包括降维，分类，协同过滤
#特征学习，和主题建模。
#RBM的R实现
library(RcppDL)
library(ltm)
#孟加拉女性社会自由流动数据：1.去过村里任何地方；2.去过村外；
#3.与陌生男人谈话；4.去影院看电影；5.购物；6.去过母亲俱乐部；
#7.出席政治会议；8.去过健康中心或者医院。
data("Mobility")
data<-Mobility

set.seed(2395)
n<-nrow(data)
sample<-sample(1:n,1000,replace = F)
data<-as.matrix(Mobility[sample,])
n<-nrow(data)
train<-sample(1:n,800,FALSE)

#数据准备
x_train<-matrix(as.numeric(unlist(data[train,])),nrow = nrow(data[train,]))
x_test<-matrix(as.numeric(unlist(data[-train,])),nrow = nrow(data[-train,]))
x_train<-x_train[,-c(4,6)]
x_test<-x_test[,-c(4,6)]
head(x_train)
head(x_test)

#构建RBM
fit<-Rrbm(x_train)
#设置隐含层三个节点的学习率为0.01
setHiddenRepresentation(fit,x=3)
setLearningRate(fit,x=0.01)

summary(fit)
#RBM和堆栈降噪自编码一样，先设置参数，然后训练
#可以在Rrbm中设置很多参数如：
#setStep signature
#setLearningRate
#setTrainingEpochs
#训练
train(fit)
#模型部署
reconProb<-reconstruct(fit,x_train)
head(reconProb,6)
#得到的是概率值
#将概率转化为二进制
recon<-ifelse(reconProb>=0.5,1,0)
head(recon)
#创建混淆矩阵
table(recon,x_train,dnn = c("Predict","Observed"))
#图像重构虽然不是RBM的重点，但是包含了RBM的主要特征
par(mfrow=c(1,2))
image(x_train,main="Train")
image(recon,main="Reconstruction")
#也可以用deepnet来评价RBM
library(deepnet)
fit2<-rbm.train(x_train,
                hidden = 3,
                numepochs = 3,
                batchsize = 100,
                learningrate = 0.8,
                learningrate_scale = 1,
                momentum = 0.5,
                visible_type = "bin",
                hidden_type = "bin",
                cd=1)
#RBM十分强大，主要用于深度置信网络的构建

#RBM由一个隐藏层和一个可见层组成，与前馈网络不同的是，RBM可见层
#和隐藏层之间的连接时无向的，并且完全连接，前提是如果允许任何层的任何神经元能够
#连接到任何层，那么就是一个玻尔兹曼机。
#RBM是一个可生成的随机神经网络，它能学习对其一组输入的概率的分布。
