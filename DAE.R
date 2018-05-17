setwd("D:/Neural Networks")
#DAE��Denoising AutoEncoders�������Ա�������AE�Ļ����ϣ���ѵ������
#������������AE�ı��壬�����Ա����������ѧϰȥ���������������û��
#���������룬��˻�ø���³���ı���
#���������ķ����������ڸ�
#DAE�������񣺶��������Խ��б��룻ȥ����������������Ӱ�죻
#��ʵ���У�һ�������Ա�������ͨ������Ա��������ø��õ����Ա��﷽����
#��ջ�����Ա������磨SDA��
#DAE��Rʵ��
#�ϼ���Ů����������������ݣ�1.ȥ�������κεط���2.ȥ�����⣻
#3.��İ������̸����4.ȥӰԺ����Ӱ��5.���6.ȥ��ĸ�׾��ֲ���
#7.��ϯ���λ��飻8.ȥ���������Ļ���ҽԺ��
#install.packages("RcppDL")
#install.packages("ltm")
library(RcppDL)
library(ltm)
data("Mobility")
data<-Mobility
#��1000����800��Ϊѵ������200��Ϊ�۲�ֵ
set.seed(17)
n<-nrow(data)
sample<-sample(1:n,1000,FALSE)
data<-as.matrix(Mobility[sample,])
n<-nrow(data)
train<-sample(1:n,800,FALSE)
x_train<-matrix(as.numeric(unlist(data[train,])),nrow = nrow(data[train,]))
x_test<-matrix(as.numeric(unlist(data[-train,])),nrow = nrow(data[-train,]))
#��Ӧ��������ΪItem3(��һ�����˽������˵��)
x_train<-x_train[,-3]
x_test<-x_test[,-3]
#��ģ
#Rsda��ģ����Ҫ��������Ӧ������������Ӧ����
y_train<-data[train,3]
temp<-ifelse(y_train==0,1,0)
y_train<-cbind(y_train,temp)

y_test<-data[-train,3]
temp1<-ifelse(y_test==0,1,0)
y_test<-cbind(y_test,temp1)

hidden<-c(10,10)
fit<-Rsda(x_train,y_train,hidden)
#Ĭ�ϵ������ȼ�Ϊ30%����Ϊ��һ�������ջ�Ա������翪ʼ��������������Ϊ0
setCorruptionLevel(fit,x=0.0)
#ע�������Rsda�������úܶ����
#setCorruptionLevel(model,x)
#����Ϊ΢����Ԥѵ���׶�ѡ������������ѧϰ��
#setFinetuneEpochs
#setFinetuneLearningRate
#setPretrainLearningRate
#setPretrainEpochs
#��ѵ����΢��ģ���൱��
pretrain(fit)
finetune(fit)
#��Ϊģ�ͺ�С�����������ܿ죬����ʹ�ò�������������һ��
predProb<-predict(fit,x_test)
head(predProb)
head(y_test,3)
#��һ������
#��������
pred1<-ifelse(predProb[,1]>=0.5,1,0)
table(pred1,y_test[,1],dnn = c("Predicted","Observed"))
#�ع�ģ�ͣ�����25%����
setCorruptionLevel(fit,x=0.25)
pretrain(fit)
finetune(fit)

predProb<-predict(fit,x_test)
pred1<-ifelse(predProb[,1]>=0.5,1,0)
table(pred1,y_test[,1],dnn = c("Predicted","Observed"))
#��������û��
#���ѧϰԭʼ���ݵĶ��ֱ�ʾ�������󣬸����ʺϸ��ӵķ�������