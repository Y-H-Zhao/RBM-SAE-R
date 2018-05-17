setwd("D:/Neural Networks")
#AE（Auto Encoder）自编码网络是一种尽可能复现输入信号的神经网络。
#为了实现这种复现，AE必须捕捉可以代表输入数据的最重要的因素，就像
#主成分。
#它是无监督的，输出误差为重构误差（输出和输入之间的误差），这之间
#通过一个编码器和一个解码器。目的是拿到编码器的输出，这样可以实现
#降维或者加密的操作。
#对于线性的AE相当于主成分的方法，而非线性的AE能够发现更复杂的主成分
#涉及笔迹和人脸识别的降维任务时，它优于主成分。
#前提：输入属性包含一些结构，即有一些关系，否则不能降维
#SAE稀疏自编码网络：在AE基础上加上L1限制，即约束每一层中节点在一次
#运算中大多数为0，即稀疏。即一些输入只刺激某些神经元，其他是抑制的。
#通过设置隐藏层神经元数目远远大于输入神经元的数目，建立输入向量x的一个
#非线性映射，并对他们实施一个稀疏约束。最受欢迎的稀疏约束是
#Kullback-Leibler散度
#SAE R实现
#install.packages("autoencoder")
library(autoencoder)
#install.packages("ripa")
library(ripa)
data(logo)
plot(logo)
#查看属性
logo
#model
x_train<-t(logo) #101行-101样本，77列-77个属性
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
###X.train=x_train图像数据传入，nl=3层数为3，unit.type = "logistic"
#使用逻辑激活函数，lambda = 1e-5权重衰减常数们通常设置为一个很小的数
#beta = 1e-5稀疏性惩罚项的权重，rho稀疏度，并按照N（0，epsilon）采样
#rescale.flag = TRUE统一重新调节训练矩阵x_train，使其值在0~1之间。
#其他参数help自行学习
#查看fit属性
attributes(fit)
#查看训练集均值误差
fit$mean.error.training.set
#模型预测
features<-predict(fit,X.input=x_train,hidden.output = TRUE)
#hidden.output = TRUE,提取隐藏节点特征（一般目的是这个）
#属性77，隐藏节点66，所以是紧凑表示
#提取隐含层节点特征
features$X.output #输出降维数据101*77变成101*60 标准化后的。
image(t(features$X.output))
#注意，使用Nelder-Mead,准线性牛顿（Quasi-Newton）,和共轭梯度算法的
#Autoencoder函数在数据包中称为优化函数，目前的优化方法包括：
#1.“BFGS”是一种拟牛顿方法，它使用函数值和梯度来建立一个图像表面进行优化
#2.“CG”是一个共轭梯度算法，主要优点是运行时不需要储存大量的矩阵。
#3.“L-BFGS-B”允许每个变量被给一个较低或者较高的限制。
#获取原来图像
pred<-predict(fit,X.input=x_train,hidden.output = FALSE)
pred$mean.error
image(t(pred$X.output))
#AE为再现，有多种变体
#Sparse AE(稀疏自编码器)
#Denoising AE(降噪自编码器)
#Regularized AE(正则自编码器)
#Contractive AE(具有惩罚项的AE)
#Marginalized DAE(边际降噪自编码器)