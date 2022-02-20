from typing import Sequence
from django.db import utils
import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.utils import np_utils
from matplotlib import pyplot as plt, units
from keras.models import model_from_json
from keras.models import load_model
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 建立簡單的線性執行的模型
model = Sequential()
# Add Input layer, 隱藏層(hidden layer) 有 256個輸出變數
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu')) 
# Add output layer
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
y_trainonehot = np_utils.to_categorical(y_train)
y_testonehot = np_utils.to_categorical(y_test)
x_train_2d = x_train.reshape(60000,28*28).astype('float32')
x_test_2d = x_test.reshape(10000,28*28).astype('float32')
x_train_norm = x_train_2d/255
x_test_norm = x_test_2d/255
train_history = model.fit(x = x_train_norm,y = y_trainonehot,validation_split=0.2,epochs=10,batch_size=800,verbose=2)
scores = model.evaluate(x_test_norm, y_testonehot)  

print()
print("\t[info] Accuracy of testing data = {:2,1f}%",format(scores[1]*100.0))
json_string = model.to_json() 
with open("model.config", "w") as text_file:    
    text_file.write(json_string)
model.save_weights("model.weight")
model.save('model.h5')