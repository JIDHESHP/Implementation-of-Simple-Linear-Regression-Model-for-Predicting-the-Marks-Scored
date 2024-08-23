# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph. 

6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JIDHESH P
RegisterNumber:  212223040078
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```


## Output:
df.head()

![image](https://github.com/user-attachments/assets/71d08e79-ad9f-4b97-99a1-471ae8f29e1a)

df.tail()

![image](https://github.com/user-attachments/assets/347540bf-9ab3-4e6e-a811-3be675f299bd)

Array value of X

![image](https://github.com/user-attachments/assets/96c79c51-37d3-4fdc-b597-67635db4e016)

Array value of Y

![image](https://github.com/user-attachments/assets/bd20acb1-85e2-431b-8044-ec9cb296da61)

Values of Y prediction

![image](https://github.com/user-attachments/assets/881008d2-1ae8-401e-a192-a60afa5e7630)

Array values of Y test

![image](https://github.com/user-attachments/assets/c9092e78-30c4-4845-9c2d-a8d99dfb5ef3)

Training Set Graph

![image](https://github.com/user-attachments/assets/28e5e46e-4d68-454a-9009-4ed5747803cb)

Test Set Graph

![image](https://github.com/user-attachments/assets/62819142-050b-453c-9be2-4f5daf9d6414)

Values of MSE, MAE and RMSE

![image](https://github.com/user-attachments/assets/a70b0aa9-c9b8-4b73-9772-9a761a91bc6c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
