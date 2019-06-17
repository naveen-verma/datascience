# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 22:48:04 2019

@author: Naveen
"""
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt

dataset = pd.read_csv('../../files/heart.csv')
print(dataset)
dataset.corr()
feature = dataset['age']
target_var = dataset['trestbps'] #resting blood pressure

def line(m,x,c):
    return (m*x+c)

def error(m,x,c,Y):
    return np.mean(line(m,x,c) - Y)**2

def derivative_slope(m,x,c, Y):
    return  2*(np.mean(line(m,x,c)-Y)*x)

def derivative_intercept(m,x,c, Y):
    return 2*(np.mean(line(m,x,c)-Y)*1)

def accuracy_for_pred(error, Y):
    return 100 - ((error/np.mean(Y**2))*100)

m = 0.2
c = 0.1
cost = []
iterations = 100
learning_rate = 0.0001
for i in range(0, iterations):
    m= m- learning_rate * derivative_intercept(m,feature,c, target_var)
    c = c - learning_rate * derivative_intercept(m,feature,c, target_var)
    cost_res = error(m,feature,c,target_var)
    print(cost_res)
    cost.append(cost_res)
    
print("Your Prediciton Accuracy: ", accuracy_for_pred(error(m,feature,c,target_var),target_var)," %")
plt.plot(cost)
plt.show()

predicted_answers = line(m,feature,c)
plt.scatter(feature,target_var)
plt.scatter(feature,predicted_answers)
plt.show()