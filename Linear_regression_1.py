# Is there any significant correlation between SAT score and GPA?


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sns.set()

data = pd.read_csv("data.csv")
print(data.describe())

y = data['GPA']
x1 = data['SAT']

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())

plt.scatter(x1,y,label='raw data')
Y = results.params[1]*x1 + results.params[0]
plt.plot(x1,Y,lw=4, c='orange',label='regression line')
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.legend()
plt.show()