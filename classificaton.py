import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("c:\\Users\\admin\\Downloads\\titanic_survivors.csv")
print(df.head(2))
print(df.info())
print(df.describe())
df.isnull().sum()
#now first of all we will do data cleaning by removing some of the null value and filling some of the null value which will be beneficial for classification later.

df=df.drop(columns=["Cabin","Name","PassengerId","Ticket"])
#now filling the null value using median.
df["Age"]=df["Age"].fillna(df.groupby("Pclass")["Age"].transform("median"))

df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Embarked"]=df["Embarked"].map({"S":0,"Q":1,"C":2})
df["Sex"]=df["Sex"].map({"male":0,"female":1})
age=df["Age"].isnull().sum()
print(df.head(5))
print(df.isnull().sum())

x=df.iloc[:,1:8]
y=df.iloc[:,0]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

log_model=LogisticRegression(max_iter=200)
log_model.fit(x_train,y_train)

y_pred=log_model.predict(x_test)
print(y_pred)

print("ACCURACY",accuracy_score(y_test,y_pred))
