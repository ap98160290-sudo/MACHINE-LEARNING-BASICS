import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


sl=pd.read_csv("C:\\Users\\admin\\Downloads\\archive (10)\\Salary_dataset.csv")
print(sl.head(2))
x=sl[["YearsExperience"]]

y=sl[["Salary"]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)

print(x_train.head(2))

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(x_train,y_train)

m=lr.coef_
print(m)
b=lr.intercept_

y_pred=lr.predict(x_test)
print(y_pred)

print(x_test.head(2))
print(y_test.head(2))
# y=mx+b
y=m*1.6+b
print(y)

new_data=[[2]]
print(lr.predict(new_data))



from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("MEAN SQUARED ERROR:",mse)
print("MEAN ABSOLUTE ERROR:",mae)
print("r2 SCORE:",r2)

plt.scatter(x_train,y_train,c="m")
plt.scatter(x_test,y_test,c="r")
plt.plot(x_train,lr.predict(x_train),c="m")
plt.xlabel("EXPERIENCE")
plt.ylabel("SALARY")
plt.show()