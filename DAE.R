setwd("D:/Neural Networks")
#DAE（Denoising AutoEncoders）降噪自编码器在AE的基础上，对训练数据
#加入噪音，是AE的变体，所以自编码网络必须学习去除这种噪音而获得没有
#噪音的输入，因此获得更加鲁棒的表达
#加入噪音的方法最常用随机掩盖
#DAE基本任务：对输入属性进行编码；去除噪音属性向量的影响；
#在实践中，一个降噪自编码网络通常会比自编码网络获得更好的属性表达方法。
#堆栈降噪自编码网络（SDA）
#DAE的R实现
#孟加拉女性社会自由流动数据：1.去过村里任何地方；2.去过村外；
#3.与陌生男人谈话；4.去影院看电影；5.购物；6.去过母亲俱乐部；
#7.出席政治会议；8.去过健康中心或者医院。
#install.packages("RcppDL")
#install.packages("ltm")
library(RcppDL)
library(ltm)
data("Mobility")
data<-Mobility
#抽1000个，800作为训练集，200作为观测值
set.seed(17)
n<-nrow(data)
sample<-sample(1:n,1000,FALSE)
data<-as.matrix(Mobility[sample,])
n<-nrow(data)
train<-sample(1:n,800,FALSE)
x_train<-matrix(as.numeric(unlist(data[train,])),nrow = nrow(data[train,]))
x_test<-matrix(as.numeric(unlist(data[-train,])),nrow = nrow(data[-train,]))
#相应变量设置为Item3(与一个不了解的男人说话)
x_train<-x_train[,-3]
x_test<-x_test[,-3]
#建模
#Rsda建模，需要用两个响应变量，构建响应变量
y_train<-data[train,3]
temp<-ifelse(y_train==0,1,0)
y_train<-cbind(y_train,temp)

y_test<-data[-train,3]
temp1<-ifelse(y_test==0,1,0)
y_test<-cbind(y_test,temp1)

hidden<-c(10,10)
fit<-Rsda(x_train,y_train,hidden)
#默认的噪声等级为30%，因为从一个常规堆栈自编码网络开始，所以噪声设置为0
setCorruptionLevel(fit,x=0.0)
#注意可以在Rsda包中设置很多参数
#setCorruptionLevel(model,x)
#可以为微调和预训练阶段选择样本数量和学习率
#setFinetuneEpochs
#setFinetuneLearningRate
#setPretrainLearningRate
#setPretrainEpochs
#欲训练和微调模型相当简单
pretrain(fit)
finetune(fit)
#因为模型很小，所以收敛很快，可以使用测试样本来检验一下
predProb<-predict(fit,x_test)
head(predProb)
head(y_test,3)
#第一个错误
#混淆矩阵
pred1<-ifelse(predProb[,1]>=0.5,1,0)
table(pred1,y_test[,1],dnn = c("Predicted","Observed"))
#重构模型，加入25%噪音
setCorruptionLevel(fit,x=0.25)
pretrain(fit)
finetune(fit)

predProb<-predict(fit,x_test)
pred1<-ifelse(predProb[,1]>=0.5,1,0)
table(pred1,y_test[,1],dnn = c("Predicted","Observed"))
#添加噪音没用
#逐层学习原始数据的多种表示，更抽象，更加适合复杂的分类任务。