# -*- coding:utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
import scipy.io as scio
import numpy as np
dataset = scio.loadmat('loadData.mat')
X = dataset['loadData'][:,:]
print(X)
Y = dataset['weight'][:,:]
print(Y)
# 定义多层感知机的网络 3个输入节点，20个隐藏节点，8个输出节点
model = Sequential()
model.add(Dense(20, input_dim=3, activation='relu'))
model.add(Dense(8, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, Y, nb_epoch=100, batch_size=10)

loss, accuracy = model.evaluate(X, Y)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

pred = model.predict(X)
accuracy = np.mean(np.argmax(pred,1) == np.argmax(Y,1))
print("Accuracy: %.2f%%" % (accuracy*100))
