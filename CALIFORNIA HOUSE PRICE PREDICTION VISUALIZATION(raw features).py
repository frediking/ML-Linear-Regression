#!/usr/bin/env python
# coding: utf-8

# In[44]:


#LOAD ALL PACKAGES & LIBRARIES
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# In[8]:


#LOAD DATASET
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()


# In[10]:


#CONVERT DATASET TO A DATA FRAME
df = pd.DataFrame(data = housing.data, columns = housing.feature_names)
df['target'] = housing.target
print(df.head())


# In[12]:


#SUMMARY STATISTICS OF DATA FRAME
print(df.describe())


# In[14]:


#DEFINE INPUT FEATURES AND OUTPUT TARGET BY DROPPING TARGET VARIABLE COLUMN FROM THE DATA FRAME
#INPUT FEATURE X
X = df.drop('target', axis = 1)

#OUTPUT TARGET VARIABLE Y
Y = df['target']


# In[16]:


#VIEW FEATURE AND TARGET SHAPE
print('Features Shape:', X.shape)
print('Target Shape:', Y.shape)


# In[18]:


#SPLITTING THE DATA (TESTING AND TRAINING)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[20]:


#TRAINING THE REGRESSION MODEL
reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)


# In[22]:


#CALCULATING PARAMETERS / COEFFICIENTS
reg.coef_


# In[24]:


#PREDITCTING THE PRICES
Yp = reg.predict(X_test)
Yp[5]


# In[26]:


# COMPARE TO Y TEST DATA
Y_test


# In[28]:


#prediction is close to the test data


# In[30]:


#MODEL EVALUATION
#FINDING THE MEAN SQUARED ERROR
mse = np.mean((Yp - Y_test)**2)
print('Mean Squared Error:', mse)


# In[52]:


#VISUALIZATIONS
plt.figure(figsize = (12, 8))
plt.scatter(Y_test, Yp, alpha = 0.5, color = 'green', label = 'data points')
plt.plot([Y_test.min(), Y_test.max()], 
         [Y_test.min(), Y_test.max()],
         'r-', 
         lw = 3, 
         label = 'Perfect Prediction Line')

plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.grid(True)
plt.show()


# In[56]:


plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches = 'tight')
import os
os.listdir()

