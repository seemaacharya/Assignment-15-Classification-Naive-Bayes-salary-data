# -*- coding: utf-8 -*-
"""
Created on Sat May 22 21:16:46 2021

@author: DELL
"""

#Importing the libraries
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Loading the dataset
salary_train = pd.read_csv("SalaryData_Train.csv")
salary_test =pd.read_csv("SalaryData_Test.csv")

#Getting the columns(considering columns without numerical)
columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

#Preprocessing
from sklearn import preprocessing
number = preprocessing.LabelEncoder() 
for i in columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])
    
#Getting the X nad y columns
colnames = salary_train.columns
len(colnames[0:13])

#Train_test_split
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
testX = salary_test[colnames[0:13]]
testY = salary_test[colnames[13]]

#Creating the Gaussian and Multinomial functions
sgnb = GaussianNB()
smnb = MultinomialNB()

#Building and predicting the model at the same time
spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
#confusion matrix(to find the accuracy)
confusion_matrix(testY,spred_gnb)
accuracy = (10759+1209)/(10759+601+2491+1209)
accuracy
#79%

spred_mnb = smnb.fit(trainX,trainY).predict(testX)
#confusion matrix(to find the accuracy)
confusion_matrix(testY,spred_mnb)
accuracy = (10891+780)/(10891+469+2920+780)
accuracy
#77%

