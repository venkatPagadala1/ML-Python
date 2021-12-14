#!/usr/bin/env python
# coding: utf-8

# # Task1:GRIP-The Sparks Foundation

# ## Data Science and Business Analytics Intern 

# ## Author: Narayana Pagadala

# ## Task: Prediction Using Supervised ML 

# ## Linear Regression with Python Scikit Learn

# #### In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables.

# ## Simple Linear Regression 

# #### In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables. 

# ## Importing Required Libraries

# In[3]:


# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ## Reading Data and Visualization 

# In[7]:


# Reading data from remote link
url = "http://bit.ly/w-data"
df = pd.read_csv(url)
df.head(10)


# In[8]:


#checking its data types
df.dtypes


# In[9]:


#checking its info
df.info()


# In[10]:


#checking its shape
df.shape


# In[11]:


#getting satistical measures of data
df.describe()


# In[12]:


#number of misssing values of each column
df.isnull().sum()


# # Plotting the distribution of scores

# In[13]:


#Plotting the distribution of scores
df.plot(x='Hours', y='Scores', style='s')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ### From above graph,we can clearly see that there is a positive linear relation between the number of hours studied and the number of percentage score 

# ## Preparing the data 

# ### The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[14]:


#separating attributes and labels
x = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values
print(x)
print(y)


# In[15]:


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 


# In[16]:


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# ## Training the Algorithm 

# In[17]:


from sklearn.linear_model import LinearRegression  
lg = LinearRegression()  


# In[18]:


lg.fit(x_train, y_train) 


# In[19]:


print('Training Completed')


# In[ ]:


lg.coef_


# In[21]:


lg.intercept_


# In[23]:


# Plotting the regression line
line = lg.coef_*x+lg.intercept_

# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line);
plt.show()


# ## Making Predictions

# In[25]:


# predicting score
y_pred = lg.predict(x_test)


# In[34]:


y_pred


# In[36]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# ## Evaluating the model

# ### The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.

# In[37]:


# You can also test with your own data
hours = 9.25
arr=np.asarray(hours)
arr_reshape=arr.reshape(1,-1)
pre=lg.predict(arr_reshape)
print('predict score for 9.25:',pre)


# In[39]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
print('MSE:',mean_absolute_error(y_test, y_pred)) #calucation of MSE
print('r2 score:', r2_score(y_test, y_pred))# calucation of r2 score

