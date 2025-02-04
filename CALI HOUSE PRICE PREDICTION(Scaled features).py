#!/usr/bin/env python
# coding: utf-8

# In[68]:


#LOAD ALL PACKAGES & LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


# In[70]:


#LOAD DATASET
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()


# In[72]:


#CONVERT DATASET TO A DATA FRAME
df = pd.DataFrame(data = housing.data, columns = housing.feature_names)
df['target'] = housing.target
print(df.head())


# In[74]:


#SUMMARY STATISTICS OF DATA FRAME
print(df.describe())


# In[76]:


#DEFINE INPUT FEATURES AND OUTPUT TARGET BY DROPPING TARGET VARIABLE COLUMN FROM THE DATA FRAME
#INPUT FEATURE X
X = df.drop('target', axis = 1)

#OUTPUT TARGET VARIABLE Y
Y = df['target']


# In[78]:


#VIEW FEATURE AND TARGET SHAPE
print('Features Shape:', X.shape)
print('Target Shape:', Y.shape)


# In[114]:


#SPLITTING THE DATA (TESTING AND TRAINING)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[116]:


#FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[118]:


#TRAINING THE REGRESSION MODEL
reg = linear_model.LinearRegression()
reg.fit(X_train_scaled, Y_train)


# In[120]:


#CALCULATING PARAMETERS / COEFFICIENTS
reg.coef_


# In[124]:


#PREDITCTING THE PRICES
Yp = reg.predict(X_test_scaled)
Yp[5]


# In[126]:


# COMPARE TO Y TEST DATA
Y_test


# In[128]:


#prediction is close to the test data


# In[130]:


#MODEL EVALUATION
#FINDING THE MEAN SQUARED ERROR
mse = np.mean((Yp - Y_test)**2)
print('Mean Squared Error:', mse)

