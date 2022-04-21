#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, Model

import matplotlib.pyplot as plt
import numpy as np



# In[5]:


SEED = 42

TRAIN_R = 0.6  # Train ratio
VAL_R = 0.2
TEST_R = 0.2

IMG_HEIGHT, IMG_WIDTH = (224, 224)
BATCH_SIZE = 32


OUTPUT_DIR = "processed_data"


train_data_dir = f"{OUTPUT_DIR}/train"
test_data_dir = f"{OUTPUT_DIR}/test"

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical")


test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=1,
    class_mode="categorical")


# In[6]:


import pandas as pd 
import seaborn as sn 
import tensorflow as tf 
model = tf.keras.models.load_model("Saved_Model\ResNet50_ton.h5")
filenames = test_generator.filenames 
#print(filenames)
nb_samples = len(test_generator) 
y_prob=[] 
y_act=[] 
test_generator.reset() 
for _ in range (nb_samples): 
    X_test,Y_test = test_generator.next() 
    y_prob.append (model.predict(X_test)) 
    y_act.append (Y_test) 
    
predicted_class = [list(train_generator.class_indices.keys())[i.argmax()] for i in y_prob] 
actual_class = [list(train_generator.class_indices.keys())[i.argmax()] for i in y_act] 

out_df = pd.DataFrame(np.vstack([predicted_class, actual_class]).T,columns=['predicted_class','actual_class']) 
confusion_matrix = pd.crosstab(out_df['actual_class'],out_df['predicted_class'], rownames=['Actual'], colnames=['Predicted']) 

sn.heatmap(confusion_matrix, cmap='Blues', annot=True, fmt='d') 
plt.show() 
print('test accuracy : {}'.format((np.diagonal (confusion_matrix).sum()/confusion_matrix.sum().sum()*100)))


# In[ ]:




