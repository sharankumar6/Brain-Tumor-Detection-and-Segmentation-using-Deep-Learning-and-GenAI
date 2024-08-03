import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# import mainTrain


model = load_model('BrainTumor_epochs_20.h5')

image=cv2.imread('D:\sharan personal\PES(M.TECH)\MTech Project\Brain_Tumor\Brain_Tumor_Tensorflow_keras\pred\pred3.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result = model.predict(input_img)
print(result)