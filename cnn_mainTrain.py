import cv2
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
import seaborn as sns
from sklearn.metrics import confusion_matrix

img_dir = 'datasets/'

#Data Extraction
no_tumor_images = os.listdir(img_dir+'no/')
yes_tumor_images = os.listdir(img_dir+'yes/')
# print(no_tumor_images)

dataset = []
label = []
INPUT_SIZE = 64

# Seperate images for 'no'
for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(img_dir+'no/'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE)) # resize the image to 64x64
        dataset.append(np.array(image))
        label.append(0)

#seperate images for 'yes'
for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(img_dir+'yes/'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

# print(len(dataset))
# print(len(label))

#convert dataset and label to numpy array
dataset = np.array(dataset)
label = np.array(label)

#Split the dataset to train and test 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)
print('Training Images:')
print(x_train.shape)
print(y_train.shape)
print('-----------------------------------')
print('Testing Images:')
print(x_test.shape)
print(y_test.shape)

#Normalize the data
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test,axis=1)

#For Categorical
y_train = to_categorical(y_train,num_classes=2)
y_test = to_categorical(y_test,num_classes=2)

#Model Building
model = Sequential()

model.add(Conv2D(32,(3,3), input_shape = (INPUT_SIZE,INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train,y_train,batch_size=16,verbose=1,epochs=20,validation_data=(x_test,y_test),shuffle=False)

model.save('BrainTumor_epochs_20_bin.h5')


score =model.evaluate(x_test, y_test)
print("%s: %2f%%" %(model.metrics_names[1],score[1]*100))

# Plotting training & validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Epochs vs. Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plotting training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Epochs vs. Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_mtx, annot=True, cmap='Blues', fmt='g')

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

