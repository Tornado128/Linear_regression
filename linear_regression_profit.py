# In this code, the profit, state of location, R&D spending, Marketing Spend and Administration costs are given in an excel file.
# We want to know:
# 1. What factor (or factors) drives the profit of the company the most?
# 2. Does state of location matter?
# 3. Can a linear regression model be predictive enough?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sns.set()

df = pd.read_csv("1000_companies.csv")
print(df.head(4))
print(df.shape)


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

# Does the amount of profit correlate with the state? We can see that the state doesn't matter
sns.barplot(x ='State', y ='Profit', data = df, palette ='plasma', estimator = np.std)
plt.show()

# mapping the name of states to numbers
df['State'] = df['State'].map({'New York': 0, 'Florida': 1, 'California':2})


# creating mask
mask = np.triu(np.ones_like(df.corr()))
# heat map (what correlates with what?)
sns.heatmap(df.corr(), cmap='spring', mask=mask)
plt.show()

# multiple linear regression:
# we can see by the P-value of state that state doesn't matter
import statsmodels.api as sm
y = df['Profit']                                                       # dependent variable
x1 = df[['R&D Spend',  'Administration', 'State' ,  'Marketing Spend']]                                       # independent variable
xx = sm.add_constant(x1)
result = sm.OLS(y,xx).fit()
print(result.summary())

# multiple linear regression
# As we saw that state doesn't matter, we don't include state in the regression
import statsmodels.api as sm
y = df['Profit']                                                                                    # dependent variable
x1 = df[['R&D Spend',  'Administration',  'Marketing Spend']]                                       # independent variable
xx = sm.add_constant(x1)
result = sm.OLS(y,xx).fit()
print(result.summary())


from sklearn.model_selection import train_test_split
x = df.iloc[:, :-2].values  # all columns except the last two (state and profit)
y = df.iloc[:, 4].values    # last column
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(regressor.coef_)
print(regressor.intercept_)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
