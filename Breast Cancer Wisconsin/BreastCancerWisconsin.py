
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers import Input, Dense
from keras.optimizers import SGD

from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd

veri = pd.read_csv("breast-cancer-wisconsin.data")

veri.replace('?', -99999, inplace= True)
#veri.drop(['id'], axis=1)
veriyeni = veri.drop(['1000025'],axis=1)

imp = Imputer(missing_values=-99999, strategy="mean",axis=0)
veriyeni = imp.fit_transform(veriyeni)


giris = veriyeni[:,0:8]
cikis = veriyeni[:,9]                   

###
#loss: 1.3715 - accuracy: 0.5573 - val_loss: 0.6489 - val_accuracy: 0.7000 [4] --sigmoid
#loss: 1.7712 - accuracy: 0.6022 - val_loss: 0.6539 - val_accuracy: 0.8714 [4] --relu
#loss: 1.2374 - accuracy: 0.7939 - val_loss: 0.2177 - val_accuracy: 0.9429 [4] --tanh
###
model = Sequential()
model.add(Dense(10, input_dim=8)) # 8 girişi oluşturduğumuz 64 nöron'lu gizli ağa bağla

model.add(Activation('relu'))     # 0 ve 1 ayırt etti
model.add(Dropout(0.5))
model.add(Dense(10))              #1. gizli katmanı 2. 32'lük gizli katmana bağladı
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))  #son kısım softmax olmalı 0 ve 1 arasına yerleştiriyor.

optimizer = keras.optimizers.SGD(learning_rate=0.01)

model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics=['accuracy']) # gerçek-tahmini'nin karesinin türevi(Gradient Descent)
model.fit(giris,cikis,epochs=100, batch_size=32,validation_split=0.20) # epochs 100 kere verisetini tarayacak 

tahmin = np.array([10,7,7,6,4,10,4,1]).reshape(1,8) # Output=2 iyi huylu Output=4 kötü huylu
print(model.predict_classes(tahmin))

