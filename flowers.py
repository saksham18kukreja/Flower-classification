# -*- coding: utf-8 -*-
"""flowers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-Et4K7t10rVWlz7PU2qglzK-KYT3z9C4
"""

from google.colab import files
files.upload()

! mkdir ~/.kaggle 
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download -d alxmamaev/flowers-recognition

!mkdir flowers
!unzip flowers-recognition.zip -d flowers

import split_folders
split_folders.ratio('flowers/flowers/flowers', output="output", seed=1337, ratio=(.8, .2))

!pip install split-folders

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
import matplotlib.pyplot as plt
from PIL import Image

model = Sequential()
model.add(Conv2D(32,(5,5),input_shape=(250,250,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(5,5),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128,(5,5),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(256,(5,5),activation='relu'))
model.add(MaxPooling2D((2,2)))



model.add(Flatten())
model.add(Dense(units=256,activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=5,activation='sigmoid'))
opt = keras.optimizers.Adam()
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])



from keras.preprocessing.image import ImageDataGenerator
training_data = ImageDataGenerator(rescale=1/255,rotation_range=15,width_shift_range=15,height_shift_range=15,horizontal_flip=True)
val_data = ImageDataGenerator(rescale=1/255)
training = training_data.flow_from_directory('output/train',target_size=(250,250),batch_size=128,class_mode='categorical')
validate = val_data.flow_from_directory('output/val',target_size=(250,250),batch_size=128,class_mode='categorical')

flowers = model.fit(training,epochs=25,validation_data=validate)

model.evaluate(validate)

from keras.preprocessing import image
test_image = image.load_img('daisy.jpg',target_size=(250,250))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
pred = np.argmax(model.predict(test_image))
print(train[pred])

acc = flowers.history['accuracy']
val_loss = flowers.history['val_loss']
val_acc = flowers.history['val_accuracy']
loss = flowers.history['loss']
plt.show()
plt.plot(acc,'b',label ='training-accuracy')
plt.plot(val_loss,'r',label='validation-loss')
plt.plot(val_acc,'black',label='validation-accuracy')
plt.plot(loss,'y',label='training-loss')
plt.legend()
plt.xlabel('epochs')
plt.title('flowers-classification')

import os
train = os.listdir('output/train')

pip install Pillow==7.0.0
