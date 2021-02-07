import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

student_dataset = pd.read_csv("F:/Python/student_scores.csv")

student_dataset.head(10)
student_dataset.describe()
student_dataset.info()
student_dataset.plot(style = "o")

#preparing the data

x = student_dataset.iloc[:,:-1]
print(x)
y = student_dataset.iloc[:,1]
print(y)


correl = student_dataset.corr()
print(correl)

#splitting training and test data

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

#train the model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)



pred = regressor.predict([[9.25]])
print("No. of Hours = 9.25")
print("Predicted Score = {}".format(pred))



