#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline

# # Reading Dataset
# url = "http://bit.ly/w-data"
# data = pd.read_csv(url)
# type(data)

# In[19]:


data.head()


# In[20]:


data.describe()


# In[21]:


data.shape


# In[22]:


data.info()


# # Relationship between the data by plottig a 2D graph
# data.plot(x= 'Hours', y= 'Scores', style='o')
# plt.title('Hours vs. Percetage')
# plt.xlabel('Hours Studied')
# plt.ylabel('Percentage score')
# plt.show()

# In[24]:


x= data.iloc[:, :-1].values
y= data.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# # Data splitted into 80% training phase and 20% test phase
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(x_train, y_train)
# 
# print("Training Complete")

# # Plotting for Rregression line and Test Data
# line = regressor.coef_*x+regressor.intercept_
# 
# plt.scatter(x,y)
# plt.plot(x, line)
# plt.show()

# In[17]:


print(x_test)
y_pred = regressor.predict(x_test)


# # Comparing actual output values for x_test with predicted values 
# data = pd.DataFrame({'Actual': y_test, 'Predicted' : y_pred})
# data

# # Testing
# hours= 9.25
# test = np.array([hours])
# test = test.reshape(-1,1)
# own_pred = regressor.predict(test)
# print("No. of Hours = {}".format(hours))
# print("Predicted Score ={}".format(own_pred[0]))

# # Evaluating
# from sklearn import metrics
# print('Mean Absolute Error:',
#      metrics.mean_absolute_error(y_test,y_pred))

# In[ ]:




