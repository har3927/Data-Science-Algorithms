#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing Required Libraries
import tensorflow as tf


# In[3]:


#Importing Required Libraries
import tensorflow as tf
#Importing the Data Set
import pandas as pd


# In[4]:


mnist=tf.keras.datasets.mnist


# In[5]:


#dividing the data set in to Traning set and Test set
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[6]:


#How Data Looks
print(x_train[0])


# In[7]:


#Visualization
import matplotlib.pyplot as plt


# In[8]:


plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()


# In[9]:


print(y_train[0])


# In[10]:


#Normalize
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)


# In[11]:


print(x_train[0])


# In[12]:


#Creating a Model
model=tf.keras.models.Sequential()


# In[13]:


model.add(tf.keras.layers.Flatten())


# In[14]:


model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))


# In[15]:


#Output Layer


# In[16]:


model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


# In[17]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[18]:


model.fit(x_train,y_train,epochs=3)


# In[19]:


val_loss,val_acc=model.evaluate(x_test,y_test)


# In[20]:


predictions=model.predict(x_test)


# In[21]:


import numpy as np
print(np.argmax(predictions[0]))


# In[22]:


plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()


# In[23]:


print(y_test[5])


# In[ ]:




