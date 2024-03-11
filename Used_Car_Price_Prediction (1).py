#!/usr/bin/env python
# coding: utf-8

# **The purpose for this code is to estimate the price of used car using linear regression technique**

# In[55]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


# In[56]:


import os
os.chdir('C:/Users/yaser/PycharmProjects/Linear_regression/Data')


# In[57]:


df_raw = pd.read_csv("raw_car_price_data.csv")


# In[58]:


## checking the first 10 rows
df_raw.head(10)


# In[59]:


df_raw.describe(include = 'all')
# some values are missing based on the first row


# In[60]:


# drop a column: Model
# axis=1 refers to the column 
df = df_raw.drop(['Model'],axis=1)


# In[61]:


df.isnull().sum()


# In[62]:


# rule of thumb: It is okay to remove rows with missing taget values as long as they are below 5% of the data
# remove the rows with missing values
df_no_mv = df.dropna(axis=0)


# In[63]:


# we see that the distribution is not normal and there are also some outliers (very expensive cars)
sns.distplot(df_no_mv['Price'])


# ##removing the first top 1% in price 

# In[64]:


q = df_no_mv['Price'].quantile(0.99)
df_1 = df_no_mv[df_no_mv['Price']<q]
df_1.describe(include='all')


# ##removing the first top 1% in mileage

# In[65]:


sns.distplot(df_1['Price'])


# In[66]:


q = df_1['Mileage'].quantile(0.99)
df_2 = df_1[df_1['Mileage']<q]
df_2.describe(include='all')


# In[67]:


sns.distplot(df_2['Mileage'])


# In[68]:


sns.distplot(df_2['EngineV'])


# In[69]:


df_3 = df_2[df_2['EngineV']<6.5]
df_3.describe(include='all')


# In[70]:


sns.distplot(df_3['Year'])


# In[71]:


q = df_3['Year'].quantile(0.01)
df_4 = df_3[df_3['Year']>q]
df_4.describe(include='all')


# In[72]:


# reseting the index from 0 to n
df_4.reset_index(drop='True')


# In[73]:


df_4.describe(include = 'all')


# In[74]:


df_4.head(20)


# In[75]:


df = df_4


# In[76]:


df.head()


# **_Can we apply linear regression?_**

# In[77]:


f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize = (15,3))
ax1.scatter(df['Year'],df['Price'])
ax1.set_title('Price and Year')
ax2.scatter(df['EngineV'],df['Price'])
ax2.set_title('Price and Engine Volume')
ax3.scatter(df['Mileage'],df['Price'])
ax3.set_title('Price and Mileage')
plt.show()


# In[78]:


log_price = np.log(df['Price']).values.reshape(-1, 1)
df['log_price'] = log_price
df.head()


# In[79]:


f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize = (15,3))
ax1.scatter(df['Year'],df['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(df['EngineV'],df['log_price'])
ax2.set_title('Log Price and Engine Volume')
ax3.scatter(df['Mileage'],df['log_price'])
ax3.set_title('Log Price and Mileage')
plt.show()


# **checking multicolinearity**

# In[80]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = df[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
vif['features'] = variables.columns


# In[81]:


vif


# In[82]:


# VIF for year is above 10. So, we can remove it safely


# In[83]:


df = df.drop(['Year'],axis=1)


# In[84]:


df.head(10)


# In[85]:


data_with_dummies = pd.get_dummies(df, drop_first = True)


# In[86]:


data_with_dummies.head(10)


# **rearrange a bit (bring log_price to the first column)**

# In[87]:


data_with_dummies.columns.values


# In[88]:


col = ['log_price', 'Price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes'
      ]


# In[89]:


data_preprocessed = data_with_dummies[col]
data_preprocessed.head()


# In[90]:


del data_preprocessed['Price']


# In[91]:


data_preprocessed.head()


# # Linear regression model

# **Declare the inputs and outputs**

# In[92]:


target = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'],axis=1)


# **scale the data**

# In[93]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)


# **train-test split**

# In[94]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, target, test_size=0.2, random_state=365)


# **create the regression**

# In[95]:


reg = LinearRegression()
reg.fit(x_train,y_train)
y_hat = reg.predict(x_train)


# In[108]:


x_train_df = pd.DataFrame(x_train)
x_train_df


# In[42]:


plt.scatter(y_train, y_hat)
plt.plot(y_train, y_train,color="red")
plt.xlabel('Target',size=18)
plt.ylabel('Predictions',size=18)
plt.xlim(6,12)
plt.ylim(6,12)
plt.show()


# In[43]:


sns.distplot(y_train-y_hat)
plt.title("residual PDF", size=18)


# In[44]:


reg.score(x_train,y_train)


# **Summary**

# In[45]:


reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary ['Weights']  = reg.coef_
reg_summary


# ## Test the model ## 

# In[46]:


y_hat_test = reg.predict(x_test)


# In[47]:


plt.scatter(y_test,y_hat_test, alpha=0.2)
plt.plot(y_test, y_test,color="red")
plt.xlabel('Target',size=18)
plt.ylabel('Predictions',size=18)
plt.xlim(6,12)
plt.ylim(6,12)
plt.show()


# In[48]:


# reseting the index (the original indicies must be omitted)
y_test = y_test.reset_index(drop=True)
y_test.head()


# In[49]:


df_performance = pd.DataFrame(np.exp(y_hat_test),columns=['Prediction'])
df_performance.head()


# In[50]:


df_performance ['Target'] = np.exp(y_test)
df_performance


# In[51]:


df_performance['Residual'] = df_performance['Target'] - df_performance['Prediction']


# In[52]:


df_performance['Difference%'] = np.abs( 100 * df_performance['Residual']/df_performance['Target'])


# In[53]:


df_performance.describe()


# In[54]:


pd.options.display.max_rows = 999
pd.set_option('display.float_format', lambda x: '%0.2f' %x)
df_performance.sort_values(by=['Difference%'])


# In[ ]:




