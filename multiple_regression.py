import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

mlr=pd.read_csv("c:\\Users\\admin\\Downloads\\archive (11)\\car data.csv")
# print(mlr.tail(10))

x=mlr.iloc[:,[1,3,4]]
y=mlr.iloc[:,2]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)

# print(x_train.head(2))


from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(x_train,y_train)
print(x_test.head(2))
print(y_test.head(2))


# new_data=[[2009,11.00,87934]]
# print(lr.predict(new_data))


y_pred=lr.predict(x_test)
print(y_pred)

m=lr.coef_
b=lr.intercept_

print(m)

y=(4.99187710e-01*2015)+(5.14739186e-01*14.79)+(-1.19483092e-06*43535+b)
print(y)