setwd("D:/Neural Networks")
#深度置信网络是由多个堆栈RBM的构成的概率生成的多层神经网络。在DBN中
#每两个相邻隐层构成一个RBM，RBM的输出就是输入的特征，每一个RBM实际上
#就是一个非线性变换，RBM的输出就是下一个RBM的输入。
#堆栈中的每一个RBM起到了对数据不同表达的作用。

#DBM训练分为两步：欲训练和不同于反向传播的微调算法。
#（1）欲训练阶段：采用逐层贪婪学习策略，独立的使用对比散度训练每个
#RBM，然后堆栈在一起。因为训练是无监督的，所以这个过程不需要标签。
#实际上，分块方法通常被用来加速预训练过程。用RBM权重修改每一个快。
#多个隐藏层产生的多个特征层越来越复杂。欲训练容易捕捉数据的高层特征。
#（2）微调：采用BP算法，自上而下的调整预训练网络的每一层的权重，
#使权重更适合分类。微调是有监督学习。
#应用案例：降维，特征提取
#DBN的R实现
#以Mobility数据集的1 2 3 5 7 8列的属性预测某人是否去看电影或
#看文化展览或去参加亲子活动或去母婴馆
#加载包
#将内存变量清空
gc(rm(list=ls()))
library(RcppDL)
library(ltm)
data("Mobility")
data<-Mobility
y<-apply(cbind(data[,4],data[.6]),1,max,na.rm=TRUE)
#数据准备
set.seed(17)
n=nrow(data)
sample<-sample(1:n,1000,FALSE)
data<-as.matrix(Mobility[sample,])
n=nrow(data)

#产生800
train<-as.integer(sample(row.names(data),800,FALSE))
#格式化训练集
y_train<-as.numeric(y[train])
temp<-ifelse(y_train==0,1,0) #取反
y_train<-cbind(y_train,temp)
head(y_train)
#格式化数据集
n1<-setdiff(sample,train) #求sample的补集 另一种方式
y_test<-as.numeric(y[n1])
temp1<-ifelse(y_test==0,1,0)
y_test<-cbind(y_test,temp1)
head(y_test)
nrow(y_train)
nrow(y_test)
#构建训练集和测试集
data<-as.data.frame(data)
x_train<-as.matrix(data[as.character(train),])
x_test<-as.matrix(data[as.character(n1),])
x_train<-x_train[,-c(4,6)]
#int to num
x_train<-as.data.frame(x_train)
x_train<-lapply(x_train,as.numeric)
x_train<-as.data.frame(x_train)
x_train<-as.matrix(x_train)

x_test<-x_test[,-c(4,6)]
#int to num
x_test<-as.data.frame(x_test)
x_test<-lapply(x_test,as.numeric)
x_test<-as.data.frame(x_test)
x_test<-as.matrix(x_test)
head(x_train)
head(x_test)
#建模
hidden=c(6,5)
fit<-Rdbn(x_train,y_train,hidden)
summary(fit)
#模型部署
#欲训练
pretrain(fit)
#微调
finetune(fit)
#用RBN模型预测测试样本分类
preProb<-predict(fit,x_test)
head(preProb,6)

pred1<-ifelse(preProb[,1]>=0.5,1,0)
table(pred1,y_test[,2],dnn = c("Predicted","Observed"))

