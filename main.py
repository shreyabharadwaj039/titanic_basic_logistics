import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

titanic_data=pd.read_csv("titanic_basic_logistics/train.csv")

# cleaning dataset

titanic_data["Age"].fillna(titanic_data["Age"].mean(),inplace=True)
titanic_data.drop("Cabin",axis=1,inplace=True)
gender=pd.get_dummies(titanic_data["Sex"],dtype=int,drop_first=True)
titanic_data["Gender"]=gender
titanic_data.drop(["Name","Ticket","Sex","Embarked"],axis=1,inplace=True)

y=titanic_data['Survived']
x=titanic_data[['PassengerId','Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
       'Gender']]

# model

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
predict=model.predict(X_test)

print(classification_report(y_test,predict))