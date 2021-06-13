from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

app=Flask(__name__)

@app.route('/')
def gets_connected():
 return(render_template('index.html'))

@app.route('/', methods=['POST'])
def read_data_form():
 pFlag=0
 name=request.form['name']
 email=request.form['email']
 rdspend=float(request.form['rdspend'])
 adminspend=float(request.form['adminspend'])
 marketspend=float(request.form['marketspend'])
 state=request.form['state']
 print(name,email,rdspend,adminspend,marketspend,state)
 if(state.lower()=='california'):
  pFlag=1
  X_sample=[[1.0,0.0,0.0,rdspend,adminspend,marketspend]]
 elif(state.lower()=='florida'):
  pFlag=1
  X_sample=[[0.0,1.0,0.0,rdspend,adminspend,marketspend]]
 elif(state.lower()=='new york'):
  pFlag=1
  X_sample=[[0.0,0.0,1.0,rdspend,adminspend,marketspend]]
 else:
  pFlag=0
 
 if(pFlag==1):
  print(X_sample)
  k=regressor.predict(X_sample)
  print(k)
 else:
  k=['No Data Available'] 
 return(render_template('index.html',prediction_output=k[0]))

if __name__=="__main__":
 app.run()
