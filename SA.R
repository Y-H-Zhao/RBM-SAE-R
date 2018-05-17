setwd("D:/Neural Networks")
#��ջ�Ա���SA��Stacked Autoencode�����磬ÿһ����һ����������
#Ϊ�˽��յı�ʾ���м���Ԫ����������Խ��ԽС��
#ÿһ������һ����ع���ʾ��������Ҫ������ȡ��ʾ������Ҫ������
#���һ���һ����ͨ������㣬���Լ򻯱�ʾ���������һ���һ�������������
# SAʵ��
#install.packages("SAENET")
library(SAENET)
aburl='http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
names=c('sex','length','diameter','height','whole.weight',
        'shuked.weight','viscera.weight','shell.weight','rings')
data=read.table(aburl,header = F,sep = ',',col.names=names)
#����׼��
#ȥ���Ա����ԣ�ɾ����������ʸߵĹ۲�ֵ�����ҽ���������洢ΪR�о������
#data1
data$sex<-NULL
data$height[data$height==0]=NA
data<-na.omit(data)
data1<-as.matrix(data)
#Ϊ��˵�����⣬��ȡ10���۲�ֵ
set.seed(2016)
n<-nrow(data)
train<-sample(1:n,10,replace = F)
#����ģ�� �������ز㣬�ڵ���n.nodes=c(5,4,2)
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
#ÿһ����������ͨ��fit[[n]]$X.output���鿴
#���������
fit[[3]]$X.output
plot(fit[[3]]$X.output[,1],fit[[3]]$X.output[,2])