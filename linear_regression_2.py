# Does GPA correlate with SAT and attendance?

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv("dummy_data.csv")                                  # read the csv file
print(df.head)
print((df['Attendance'].unique()))                                  # unique values in 'Attendance' column
print(len(df['Attendance'].unique()))                               # number of unique values in 'Attendance' column
df['Attendance'] = df['Attendance'].map({'Yes': 1, 'No': 0})        # replace 'Yes' with 1 and 'No' with 0
print(df.describe())

# multiple linear regression
y = df['GPA']                                                       # dependent variable
x1 = df[['SAT','Attendance']]                                       # independent variable

x = sm.add_constant(x1)                                             #
result = sm.OLS(y,x).fit()
print(result.summary())

Y_yes = 0.0014 * df['SAT'] + 0.6439 + 0.226
Y_no = 0.0014 * df['SAT'] + 0.6439

plt.plot(df['SAT'],Y_yes, c='#006837')
plt.plot(df['SAT'],Y_no, c='#a50026')
plt.legend(['attended','not attended'])
plt.scatter(df['SAT'],df['GPA'], c=df['Attendance'], cmap='RdYlGn')
plt.xlabel("SAT grade",fontsize=20)
plt.ylabel("GPA",fontsize=20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()

new_data = pd.DataFrame({'const':1, 'SAT':[1700,1670],'Attendance':[0,1]})
predictions = result.predict(new_data)
print(predictions)
