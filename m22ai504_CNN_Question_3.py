#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,GlobalAveragePooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


# Define the paths to your image and csv folders
train_val_dir = "C:\\Users\\Aditya\\Desktop\\iitj\\machine learning sem 2\\ML Assignment Fractal 3\\Q3\\charts\\train_val"
test_dir = "C:\\Users\\Aditya\\Desktop\\iitj\\machine learning sem 2\\ML Assignment Fractal 3\\Q3\\charts\\test"
train_path_labels = "C:\\Users\\Aditya\\Desktop\\iitj\\machine learning sem 2\\ML Assignment Fractal 3\\Q3\\charts\\train_val.csv"
train_val_labels = pd.read_csv(train_path_labels)
print(pd.read_csv(train_path_labels))


# In[31]:


# load training dataset in numpy array
images = []
labels = []

for filename in os.listdir(train_val_dir):
    if filename.endswith('.png'):
        # Load the images and resize them to (128, 128)
        image = cv2.imread(os.path.join(train_val_dir, filename))
        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # image = Image.open(os.path.join(train_val_dir, filename))
        image_array = np.array(image)

        # Append the array to the list of images
        images.append(image_array)
        labels.append(filename)

lcoder = LabelEncoder()
lables = lcoder.fit_transform(labels)

images = np.array(images)
labels = np.array(labels)

np.save('x_train.npy', images)
np.save('y_train.npy', labels)
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')


# In[4]:


x_train.shape


# In[5]:


# load test dataset in numpy array
images = []
labels = []
for filename in os.listdir(test_dir):
    if filename.endswith('.png'):
        # Load the images and resize them to (128, 128) 
        image = cv2.imread(os.path.join(test_dir, filename))
        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.open(os.path.join(test_dir, filename))
        image_array = np.array(image)
        # Append the array to the list of images
        images.append(image_array)
        labels.append(filename)

lcoder = LabelEncoder()
labels = lcoder.fit_transform(labels)

images = np.array(images)
labels = np.array(labels)

np.save('x_test.npy', images)
np.save('y_test.npy', labels)
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')


# In[6]:


x_test.shape


# In[25]:


# check the images loaded
plt.figure(figsize = (15,5))
plt.imshow(x_train[10])
plt.imshow(x_train[200])
plt.imshow(x_train[445])


# In[8]:


# define some classes from the images we have observed
image_cls = ['line', 'dot_line', 'hbar_categorical', 'vbar_categorical', 'pie']
image_cls[0]

label_map = {'line': 0, 'dot_line': 1, 'hbar_categorical': 2, 'vbar_categorical': 3, 'pie': 4}
y_train = np.array([label_map[label] for label in train_val_labels['type']])
y_train
y_train.shape
y_test.shape


# In[9]:


# map the lables from csv to the images
def image_sample(x, y, index):
    plt.figure(figsize = (10,2))
    plt.imshow(x[index])
    plt.xlabel(image_cls[y[index]])


# In[26]:


image_sample(x_train,y_train,0)
image_sample(x_train,y_train,200)
image_sample(x_train,y_train,445)


# In[27]:


# normalize the image
x_train=x_train /255
x_test=x_train /255
x_test.shape


# In[28]:


y_train_index = train_val_labels['image_index']
y_train_type = train_val_labels['type']
y_train_type[:5]


# In[29]:


model = Sequential([
Flatten(input_shape=(128,128,3)),
Dense(3000, activation='relu'),
Dense(1000, activation='relu'),
Dense(5, activation='softmax')
])
# Compile the model
model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10)


# In[15]:


print("Train Images Shape:", x_train.shape)
print("Train Labels Shape:", y_train.shape)
print("Test Images Shape:", x_test.shape)
print("Test Labels Shape:", y_test.shape)


# In[16]:


# modify the model architecture to cnn
cnn_model = Sequential([
    Conv2D(filters=16 ,kernel_size=(3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])
# Compile the model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
history = cnn_model.fit(x_train, y_train, batch_size=1000, epochs=50,validation_data=(x_test, y_test))

# Plot the obtained loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# In[ ]:


cnn_model.evaluate(x_test,y_test)


# In[64]:


image_sample(x_test,y_test,1)
image_sample(x_test,y_test,50)
image_sample(x_test,y_test,25)
image_sample(x_test,y_test,30)


# In[38]:


y_pred = cnn_model.predict(x_test)
y_pred[:5]


# In[39]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[40]:


y_test[:5]


# In[41]:


image_sample(x_test,y_test,15) #actual
image_classes[y_classes[15]] #predicted


# In[42]:


print("classification report: \n", classification_report(y_test,y_classes))


# In[32]:


from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Load the pre-trained model
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[33]:


# Replace the final classification layer with a new layer
x = vgg16_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)
pt_model = tf.keras.Model(inputs=vgg16_model.input, outputs=predictions)


# In[34]:


# Freeze the weights of all layers except the new classification layer
for layer in pt_model.layers:
    layer.trainable = False


# In[35]:


# Compile the model with categorical crossentropy loss and Adam optimizer
pt_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[36]:


# Print the summary of the model architecture
pt_model.summary()


# In[37]:


# Set up data generators for image augmentation and feeding data to the model
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)


# In[38]:


# flow method generates batches of augmented data
train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
test_generator = train_datagen.flow(x_test, y_test, batch_size=32)


# In[60]:


# Train the model with early stopping
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
history = pt_model.fit(train_generator, epochs=100, validation_data=test_generator, callbacks=[es])


# In[24]:


# AlexNet
input_shape = (227, 227, 3)

alexNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(384, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(384, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])


# In[62]:


alexNet.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[65]:


alexNet.summary()


# In[67]:


alexNet.fit(x_train, y_train_onehot, epochs=10, batch_size=32)


# In[66]:


y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=5)


# In[ ]:




