#!/usr/bin/env python
# coding: utf-8

# In this task we will learn about **Linear Regression** and also work on a small project.

# # Linear Regression

# Linear regression is a supervised learining algorithm used to predict the value of a variable based on the value of another variable. The variable you want to predict is called the dependent variable. The variable you are using to predict the other variable's value is called the independent variable.

# # Simple Linear Regression
# 
# Simple linear regression is a regression model that estimates the relationship between one independent variable and one dependent variable using a straight line.

# # Task 1: Prediction Using Supervised ML
# 
# ## AIM:
#     Predict the percentage of an student based on no of study hours.

# ### STEP 1: Importing Required Libraries

# In[1]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# ### STEP 2: Reading the Dataset

# In[2]:


# Reading data from remote link
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")


# In[3]:


data


# In[21]:


data.describe()


# In[22]:


data.head(5) # gives the first 5rows of the dataset


# ### STEP 3: Visualising the Data

# In[24]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', color='red', style='x')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ### STEP 4: Prepring the Data

# In[7]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values


# In[8]:


X


# In[9]:


y


# In[10]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# ### STEP 5: Training the Algorithm

# In[11]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# ### STEP 6: Visualising The Model

# In[27]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the train data
plt.scatter(X, y, color='red')
plt.plot(X, line, color='green')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()


# In[28]:


# Plotting for the test data
plt.scatter(X_test, y_test, color='red')
plt.plot(X, line, color='green')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.grid()
plt.show()


# ### STEP 6: Making Predictions

# In[15]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[16]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[29]:


# You can also test with your own data
hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ### STEP 7: Evaluating the Model

# In[19]:


### **Evaluating the model**
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# ## Conclusion

# Hence, it is concluded that the percentage if a person studies for 9.25 hours is **93.69173248737538**

# In[ ]:




