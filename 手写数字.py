import numpy
from keras.datasets import mnist # 从keras的datasets中导入mnist数据集
from keras.models import Sequential # 导入Sequential模型
from keras.layers import Dense # 全连接层用Dense类
from keras.layers import Dropout # 为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合
from keras.utils import np_utils # 导入np_utils是为了用one hot encoding方法将输出标签的向量（vector）转化为只在出现对应标签的那一列为1，其余为0的布尔矩阵

seed = 7 #设置随机种子
numpy.random.seed(seed)
(X_train,y_train),(X_test,y_test) = mnist.load_data() #加载数据
#print(X_train.shape[0])
#数据集是3维的向量（instance length,width,height).对于多层感知机，模型的输入是二维的向量，因此这里需要将数据集reshape，即将28*28的向量转成784长度的数组。可以用numpy的reshape函数轻松实现这个过程。
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0],num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0],num_pixels).astype('float32')

#给定的像素的灰度值在0-255，为了使模型的训练效果更好，通常将数值归一化映射到0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# 搭建神经网络模型了，创建一个函数，建立含有一个隐层的神经网络
def baseline_model():
    model = Sequential() # 建立一个Sequential模型,然后一层一层加入神经元
    # 第一步是确定输入层的数目正确：在创建模型时用input_dim参数确定。例如，有784个个输入变量，就设成num_pixels。
    #全连接层用Dense类定义：第一个参数是本层神经元个数，然后是初始化方式和激活函数。这里的初始化方法是0到0.05的连续型均匀分布（uniform），Keras的默认方法也是这个。也可以用高斯分布进行初始化（normal）。
    # 具体定义参考：https://cnbeining.github.io/deep-learning-with-python-cn/3-multi-layer-perceptrons/ch7-develop-your-first-neural-network-with-keras.html
    model.add(Dense(num_pixels,input_dim=num_pixels,kernel_initializer='normal',activation='relu'))
    model.add(Dense(num_classes,kernel_initializer='normal',activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model = baseline_model()
#model.fit() 函数每个参数的意义参考：https://blog.csdn.net/a1111h/article/details/82148497
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=200,verbose=2)
# 1、模型概括打印
model.summary()

scores = model.evaluate(X_test,y_test,verbose=0) #model.evaluate 返回计算误差和准确率
print(scores)
print("Base Error:%.2f%%"%(100-scores[1]*100))