#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
from keras.layers import Dense, Flatten


# In[2]:


# Access the folder
train = 'C:\\Users\\Aditya\\Desktop\\iitj\\machine learning sem 2\\ML Assignment Fractal 3\\Q2'
val = 'C:\\Users\\Aditya\\Desktop\\iitj\\machine learning sem 2\\ML Assignment Fractal 3\\Q2'
data = train


# In[8]:


# Define image size
image_size = (32, 32)

# Create lists for the images and labels
images = []
labels = []
# Loop over each folder
for label in range(10):
    folder = os.path.join(data, 'train', str(label))
    # Loop over each image 
    for file in os.listdir(folder):
        file_loc = os.path.join(folder, file)
        if file_loc.endswith(('.tiff','.bmp')):
            # Load the image and resize
            image = cv2.imread(file_loc, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(img, image_size)
           
            images.append(image)
            labels.append(label)
            
# Save the arrays
np.save('x_train.npy', images)
np.save('y_train.npy', labels)


# In[10]:


# Set the path to the folder containing the 'val' 
data_d = val

# Set the image size
image_size_val = (32, 32)

# Create empty lists for the images and labels
images_val = []
labels_val = []

# Loop over each folder
for label in range(10):
     folder = os.path.join(data_d, 'val\\', str(label))

     
     for file in os.listdir(folder):
        file_loc = os.path.join(folder, file)
        if file_loc.endswith(('.tiff','.bmp')):
            image = cv2.imread(file_loc, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(img, image_size_val)
            images_val.append(image)
            labels_val.append(label)
                    
images_val = np.array(images_val)
labels_val = np.array(labels_val)

np.save('x_test.npy', images_val)
np.save('y_test.npy', labels_val)


# In[11]:


# Load the dataset
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')


# In[12]:


# test the images are loaded correctly
print(len(x_train))
print(len(x_test))
x_train[0].shape
x_train[0]
plt.matshow(x_train[0])
plt.matshow(x_train[999])
print(x_train.shape)
print(x_test.shape)
y_train
y_test
plt.matshow(x_test[150])


# In[13]:


# creating a simple nn
model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(10, input_shape=(1024,), activation='sigmoid')
])

# compile the nn
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


# In[14]:


# scale and try to check the accuracy
x_train_scaled = x_train/255
x_test_scaled = x_test/255
model.fit(x_train_scaled, y_train,epochs= 10, validation_data=(x_test_scaled, y_test))


# In[15]:


# evaluate test dataset
model.evaluate(x_test_scaled,y_test)


# In[23]:


# predict 2st image
plt.matshow(x_test[1])
y_predicted = model.predict(x_test_scaled)
y_predicted[1]
print('Predicted Value is ',np.argmax(y_predicted[0]))
plt.matshow(x_test[88])
print('Predicted Value is ',np.argmax(y_predicted[88]))
plt.matshow(x_test[177])
print('Predicted Value is ',np.argmax(y_predicted[177]))


# In[24]:


y_predicted_labels=[np.argmax(i) for i in y_predicted]
print(y_predicted_labels, len(y_predicted_labels))
conf_mt = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
conf_mt


# In[25]:


import seaborn as sn
plt.figure(figsize = (5,5))
sn.heatmap(conf_mt,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[22]:


# adding more layers for accuracy
model2 = keras.Sequential([
 keras.layers.Flatten(),
 keras.layers.Dense(1024,input_shape=(1024,), activation='relu'),
 keras.layers.Dense(10, activation='softmax')
])
# compile the nn
model2.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy']
 )
# train the model
history = model2.fit(x_train_scaled, y_train,epochs= 10, validation_data=(x_test_scaled, y_test))


# In[66]:


# evaluate test dataset on modified model
model2.evaluate(x_test_scaled,y_test)


# In[67]:


y_predicted = model2.predict(x_test_scaled)
y_predicted[0]
y_predicted_labels=[np.argmax(i) for i in y_predicted]
print(y_predicted_labels, len(y_predicted_labels))
conf_mt = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
conf_mt


# In[68]:


plt.figure(figsize = (5,5))
sn.heatmap(conf_mt,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[69]:


# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()





