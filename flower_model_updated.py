# kaggle datasets download -d alxmamaev/flowers-recognition

import split_folders
split_folders.ratio('flowers/flowers/flowers', output="output", seed=1337, ratio=(.8, .2))

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
from keras.layers import BatchNormalization
import os
import cv2
from keras.applications import ResNet50


# defining a CNN-model 
def scratch_model():
  model = Sequential()
  model.add(Conv2D(32,(3,3),input_shape=(256,256,3),padding='same',activation='relu'))
  model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
  model.add(MaxPooling2D((2,2)))


  model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
  model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
  model.add(MaxPooling2D((2,2)))


  model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
  model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
  model.add(MaxPooling2D((2,2)))


  model.add(Conv2D(256,(3,3),activation='relu'))
  model.add(Dropout(0.5))
  model.add(MaxPooling2D((2,2)))


  model.add(Flatten())
  model.add(Dense(units=512,activation='relu'))
  model.add(BatchNormalization())
  model.add(Dense(units=256,activation='relu'))
  model.add(BatchNormalization())
  model.add(Dense(units=5,activation='softmax'))
  opt = keras.optimizers.Adam()
  model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

  return model


#defining a ResNet50 model
def model_from_ResNet():
    model = Sequential()

    model.add(ResNet50(include_top = True,pooling='max', weights='imagenet',input_shape=(224,224,3)))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(len(categories), activation='softmax'))

    model.layers[0].trainable = False
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # optimizer=RMSprop(lr=0.001)

    return model



#preprocessing the data
from keras.preprocessing.image import ImageDataGenerator
training_data = ImageDataGenerator(rescale=1/255,rotation_range=15,width_shift_range=15,height_shift_range=15,horizontal_flip=True)
val_data = ImageDataGenerator(rescale=1/255)
training = training_data.flow_from_directory('output/train',target_size=(256,256),batch_size=64,class_mode='categorical')
validate = val_data.flow_from_directory('output/val',target_size=(256,256),batch_size=64,class_mode='categorical')



# function to run model
def run_model(model):
  flowers = model.fit(training,epochs=25,validation_data=validate)


model1 = scratch_model()
scratch = run_model(model1)

model2 = model_from_ResNet()
resnet_model = run_model(model2)

#function for the loss and accuracy graphs of the model
def model_graphs(model):
  acc = model.history['accuracy']
  val_loss = model.history['val_loss']
  val_acc = model.history['val_accuracy']
  loss = model.history['loss']
  plt.imshow()
  fig,ax = plt.subplot(2)
  ax[0].plot(acc,'blue',label ='training-accuracy')
  ax[0].plot(val_acc,'black',label='validation-accuracy')
  plt.show()
  ax[1].plot(loss,'yellow',label='training-loss')
  ax[1].plot(val_loss,'red',label='validation-loss')
  plt.legend()
  plt.xlabel('epochs')
  plt.title('flowers-classification %s'%(model))

model_graphs(scratch_model)
model_graphs(resnet_model)


#function to predict the class of an image
def predict_image(file,model):
  train = os.listdir('output/train')
  from keras.preprocessing import image
  test_image = image.load_img(file,target_size=(256,256))
  test_image = image.img_to_array(test_image)
  test_image = np.expand_dims(test_image,axis=0)
  test_image/=255
  pred = model.predict(test_image)
  return train[pred]

file = cv2.imread('Daisy.jpg')
prediction = predict_image(file,model1)
plt.imshow(file)
print('the model predicted a %s'%(prediction))






