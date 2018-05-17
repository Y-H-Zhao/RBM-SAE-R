setwd("D:/Neural Networks")
#堆栈自编码SA（Stacked Autoencode）网络，每一层是一个编码器，
#为了紧凑的表示，中间神经元的数量往往越变越小。
#每一层是上一层的重构表示，或者重要特征提取表示。不需要解码器
#最后一层接一个普通的输出层，可以简化表示。或者最后一层接一个深度置信网络
# SA实现
#install.packages("SAENET")
library(SAENET)
aburl='http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
names=c('sex','length','diameter','height','whole.weight',
        'shuked.weight','viscera.weight','shell.weight','rings')
data=read.table(aburl,header = F,sep = ',',col.names=names)
#数据准备
#去掉性别属性，删除编码出错率高的观测值，并且将结果样本存储为R中矩阵对象
#data1
data$sex<-NULL
data$height[data$height==0]=NA
data<-na.omit(data)
data1<-as.matrix(data)
#为了说明问题，抽取10个观测值
set.seed(2016)
n<-nrow(data)
train<-sample(1:n,10,replace = F)
#建立模型 三个隐藏层，节点数n.nodes=c(5,4,2)
fit<-SAENET.train(X.train = data1[train,],
                  n.nodes = c(5,4,2),
                  unit.type = "logistic",
                  lambda = 1e-5,
                  beta = 1e-5,
                  rho = 0.07,
                  epsilon = 0.1,
                  max.iterations = 100,
                  optim.method = c("BFGS"),
                  rel.tol = 0.01,
                  rescale.flag = TRUE,
                  rescaling.offset = 0.001)
#每一层的输出可以通过fit[[n]]$X.output来查看
#第三层二个
fit[[3]]$X.output
plot(fit[[3]]$X.output[,1],fit[[3]]$X.output[,2])
