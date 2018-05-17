setwd("D:/Neural Networks")
#AE��Auto Encoder���Ա���������һ�־����ܸ��������źŵ������硣
#Ϊ��ʵ�����ָ��֣�AE���벶׽���Դ����������ݵ�����Ҫ�����أ�����
#���ɷ֡�
#�����޼ල�ģ�������Ϊ�ع������������֮���������֮��
#ͨ��һ����������һ����������Ŀ�����õ����������������������ʵ��
#��ά���߼��ܵĲ�����
#�������Ե�AE�൱�����ɷֵķ������������Ե�AE�ܹ����ָ����ӵ����ɷ�
#�漰�ʼ�������ʶ��Ľ�ά����ʱ�����������ɷ֡�
#ǰ�᣺�������԰���һЩ�ṹ������һЩ��ϵ�������ܽ�ά
#SAEϡ���Ա������磺��AE�����ϼ���L1���ƣ���Լ��ÿһ���нڵ���һ��
#�����д����Ϊ0����ϡ�衣��һЩ����ֻ�̼�ĳЩ��Ԫ�����������Ƶġ�
#ͨ���������ز���Ԫ��ĿԶԶ����������Ԫ����Ŀ��������������x��һ��
#������ӳ�䣬��������ʵʩһ��ϡ��Լ�������ܻ�ӭ��ϡ��Լ����
#Kullback-Leiblerɢ��
#SAE Rʵ��
#install.packages("autoencoder")
library(autoencoder)
#install.packages("ripa")
library(ripa)
data(logo)
plot(logo)
#�鿴����
logo
#model
x_train<-t(logo) #101��-101������77��-77������
set.seed(2016)
fit<-autoencode(X.train=x_train,X.test = NULL,
                nl=3,N.hidden = 60,
                unit.type = "logistic",
                lambda = 1e-5,beta = 1e-5,
                rho = 0.3,epsilon = 0.1,
                max.iterations = 100,
                optim.method = c("BFGS"),
                rel.tol = 0.01,rescale.flag = TRUE,
                rescaling.offset = 0.001)
###X.train=x_trainͼ�����ݴ��룬nl=3����Ϊ3��unit.type = "logistic"
#ʹ���߼��������lambda = 1e-5Ȩ��˥��������ͨ������Ϊһ����С����
#beta = 1e-5ϡ���Գͷ����Ȩ�أ�rhoϡ��ȣ�������N��0��epsilon������
#rescale.flag = TRUEͳһ���µ���ѵ������x_train��ʹ��ֵ��0~1֮�䡣
#��������help����ѧϰ
#�鿴fit����
attributes(fit)
#�鿴ѵ������ֵ���
fit$mean.error.training.set
#ģ��Ԥ��
features<-predict(fit,X.input=x_train,hidden.output = TRUE)
#hidden.output = TRUE,��ȡ���ؽڵ�������һ��Ŀ���������
#����77�����ؽڵ�66�������ǽ��ձ�ʾ
#��ȡ������ڵ�����
features$X.output #�����ά����101*77���101*60 ��׼����ġ�
image(t(features$X.output))
#ע�⣬ʹ��Nelder-Mead,׼����ţ�٣�Quasi-Newton��,�͹����ݶ��㷨��
#Autoencoder���������ݰ��г�Ϊ�Ż�������Ŀǰ���Ż�����������
#1.��BFGS����һ����ţ�ٷ�������ʹ�ú���ֵ���ݶ�������һ��ͼ���������Ż�
#2.��CG����һ�������ݶ��㷨����Ҫ�ŵ�������ʱ����Ҫ��������ľ���
#3.��L-BFGS-B������ÿ����������һ���ϵͻ��߽ϸߵ����ơ�
#��ȡԭ��ͼ��
pred<-predict(fit,X.input=x_train,hidden.output = FALSE)
pred$mean.error
image(t(pred$X.output))
#AEΪ���֣��ж��ֱ���
#Sparse AE(ϡ���Ա�����)
#Denoising AE(�����Ա�����)
#Regularized AE(�����Ա�����)
#Contractive AE(���гͷ����AE)
#Marginalized DAE(�߼ʽ����Ա�����)