#Slip 2
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
data_set=pd.read_csv('/home/dell12/Desktop/DataAnalytics/salary1.csv')
x=data_set.iloc[:,1:2].values
y=data_set.iloc[:,2].values
print(data_set.head())
print(x)
print(y)
from sklearn.linear_model import LinearRegression
lin_regs=LinearRegression()
lin_regs.fit(x,y)
LinearRegression(copy_X=True,fit_intercept=True,n_jobs=None,normalize=False)
mtp.scatter(x,y,color="blue")
mtp.plot(x,lin_regs.predict(x),color="red")
mtp.title("salary estimation model using Linear Regression")
mtp.xlabel("position levels")
mtp.ylabel("salary")
mtp.show()
