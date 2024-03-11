# The goal is to predict the price of car
# The data is obtained from Kaggle
## Hint: There is something wrong with the dataset in the sense that the model does not match with the make

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sns.set()

df = pd.read_csv("car_price_predictions.csv")
print(df.head(10))
df = df.iloc[:,1:]                                                       # We don't need the first column
print(df.shape)

#print(df.isnull())                                                      # we don't have null in this dataset

print((df['Make'].unique()))                                             # unique values in 'Make' column
print((df['Model'].unique()))                                            # unique values in 'Model' column
print((df['Condition'].unique()))                                        # unique values in 'Condition' column

print(df.groupby(['Make'])['Price'].mean())
print(df.groupby(['Model'])['Price'].mean())
##from sklearn.preprocessing import LabelEncoder, OneHotEncoder
##from sklearn.compose import ColumnTransformer
##labelencoder = LabelEncoder()
##x[:, 3] = labelencoder.fit_transform(x[:, 3])
## Specify the columns to be one-hot encoded using ColumnTransformer
##column_transformer = ColumnTransformer(
##    transformers=[
##        ('onehot', OneHotEncoder(), [3])  # 3 is the index of the column to be one-hot encoded
##    ],
##    remainder='passthrough'
##)
##x = column_transformer.fit_transform(x)

# Does price correlate with "Make"?

fig, (ax1, ax2) = plt.subplots(1,2)
sns.barplot(x ='Make', y ='Price', data = df, palette ='plasma', estimator = np.std, ax=ax1)
sns.barplot(x ='Model', y ='Price', data = df, palette ='plasma', estimator = np.std, ax=ax2)
plt.show()

df['Make'] = pd.factorize(df['Make'], sort=True)[0]
df['Model'] = pd.factorize(df['Model'], sort=True)[0]
df['Condition'] = pd.factorize(df['Condition'], sort=True)[0]

# creating mask
mask = np.triu(np.ones_like(df.corr()))
# heat map (what correlates with what?)
sns.heatmap(df.corr(), cmap='spring', mask=mask)
plt.show()

# As we are seeing that Price correlates significantly with year and mileage and to some extent with condition, we don't consider the Model
# and make in the regression
import statsmodels.api as sm
y = df['Price']                                                       # dependent variable
x1 = df[['Year' , 'Mileage']]                                       # independent variable
xx = sm.add_constant(x1)
result = sm.OLS(y,xx).fit()
print(result.summary())

#Model vs Prediction uisng statsmodel
Y_price = -999.9994 * df['Year'] - 0.0500 * df['Mileage'] + 2.042e+06
plt.plot(df['Price'],Y_price, 'o', c='#006837')
plt.xlabel("Data",fontsize=20)
plt.ylabel("Prediction",fontsize=20)
plt.show()

# model vs Prediction using sklearn
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x1, y)
Y_price = regressor.predict(x1)
print(regressor.coef_)
print(regressor.intercept_)