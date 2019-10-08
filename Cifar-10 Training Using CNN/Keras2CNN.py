# -*- coding: utf-8 -*-
from scipy import ndimage
from scipy import misc
import numpy
from matplotlib import pyplot
from scipy.misc import toimage
import scipy.misc

from keras.datasets import cifar10 # 10 class dan oluşan 60000'lik veri seti
from keras.models import Sequential
from keras.layers import Dense   # Düğüm sayısına göre yeni katman yaratılır.
from keras.layers import Dropout # Eğitim aşamasında overfitting i azaltmak için kullanılır. 
from keras.layers import Flatten # Matris formundaki veriyi düzleştirmek için kullanılır.
import keras.layers
from keras.constraints import maxnorm # Weight constraint.
from keras.optimizers import SGD # stochastic gradient descent optimizasyon algoritması.
from keras.layers.convolutional import Conv2D # İki boyultlu resimde rastgele filtreleme.
from keras.layers.convolutional import MaxPooling2D # Detay kaybetmeden oluşturulan kernel'a göre en büyük değerleri alıp boyut düşürme
from keras.utils import np_utils
from keras import backend as K
keras.backend.image_data_format() # cifar10 indirecek




(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# normalize inputs from 0-255 to 0.0-1.0
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

for i in range(0, 9):
 pyplot.subplot(330 + 1 + i)
 pyplot.imshow(toimage(X_train[i]))
# show the plot
pyplot.show()

cat = ndimage.imread("C://Users//atakb//Desktop//Keras//car.jpg")
cat = scipy.misc.imresize(cat,(32,32)) # Yeniden boyutlandır.
cat = numpy.array(cat)
print("Resim Okundu!")
cat = cat.reshape(1,32,32,3)

model = Sequential()
# 32 farklı 3*3'lük kernel üretti. #input_shape(32, 32, 3)--> Resmin boyutları 32*32. 3 Kanallı(RGB)
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
# MaxPooling yaparak boyut 16*16 oldu
model.add(MaxPooling2D(pool_size=(2, 2)))
# 3*3 filter 
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# MaxPooling yaparak boyut 8*8 oldu
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# MaxPooling yaparak boyut 4*4 oldu
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# MaxPooling yaparak boyut 2*2 oldu
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten --> 3*32*32 --> matris düzleştirme
model.add(Flatten())
model.add(Dense(1000, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) # 10 input vardı -- 10 output olucak

# Compile Model
epochs = 10 # cycle
lrate = 0.001 # learning rate
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary()) # Model'in özetini çıkartır

# validation_split=0.2 -->> test için ayrılan oran.
model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

print("---------------")

print(model.predict_classes(cat))
