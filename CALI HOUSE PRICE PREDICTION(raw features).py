#!/usr/bin/env python
# coding: utf-8

# In[50]:


#LOAD ALL PACKAGES & LIBRARIES
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[52]:


#LOAD DATASET
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()


# In[54]:


#CONVERT DATASET TO A DATA FRAME
df = pd.DataFrame(data = housing.data, columns = housing.feature_names)
df['target'] = housing.target
print(df.head())


# In[18]:


#SUMMARY STATISTICS OF DATA FRAME
print(df.describe())


# In[56]:


#DEFINE INPUT FEATURES AND OUTPUT TARGET BY DROPPING TARGET VARIABLE COLUMN FROM THE DATA FRAME
#INPUT FEATURE X
X = df.drop('target', axis = 1)

#OUTPUT TARGET VARIABLE Y
Y = df['target']


# In[58]:


#VIEW FEATURE AND TARGET SHAPE
print('Features Shape:', X.shape)
print('Target Shape:', Y.shape)


# In[60]:


#SPLITTING THE DATA (TESTING AND TRAINING)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[62]:


#TRAINING THE REGRESSION MODEL
reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)


# In[64]:


#CALCULATING PARAMETERS / COEFFICIENTS
reg.coef_


# In[66]:


#PREDITCTING THE PRICES
Yp = reg.predict(X_test)
Yp[5]


# In[68]:


# COMPARE TO Y TEST DATA
Y_test


# In[42]:


#prediction is close to the test data


# In[70]:


#MODEL EVALUATION
#FINDING THE MEAN SQUARED ERROR
mse = np.mean((Yp - Y_test)**2)
print('Mean Squared Error:', mse)


# In[ ]:




