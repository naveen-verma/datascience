import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as random

data = pd.read_csv("../../files/housing.csv")
# data.columns
target_var = data["median_house_value"]
target_var = np.array(target_var)
target_var = np.reshape(target_var,[1, len(target_var)])

features = data.drop(["median_house_value"], axis=1).drop(["ocean_proximity"],axis=1)
features = np.array(features)

#print(target_var)
#print(features)
# print("target_var shape: ", target_var.shape)
# print("features shape: ", features.shape)

def line_multidim(m,x,c):
    return np.dot(m,x.T) + c

def error(m,x,c,y):
    return np.mean((line_multidim(m,x,c)-y)**2)
    
def derivative_slopes(m,x,c,y):
    return 2 * np.mean(np.multiply((line_multidim(m,x,c) - y), x.T), axis = 1)

def derivative_intercept(m,x,c,y):
    return 2 * np.mean((line_multidim(m,x,c)-y))

def accuracy_pred(error, y):
    return 100 -  (error/np.mean(y**2))*100
#print(line_multidim(m, features,c))
#print(derivative_intercept(m,features,c,target_var))
m = np.random.randn(1,8)
c = random.random()

iterations = 1000
lr = 0.0001
error_array = []
for i in range(0, iterations):
    m = m-lr*derivative_slopes(m,features,c,target_var)
    c = c-lr*derivative_intercept(m,features,c,target_var)
    error_array.append(error(m,features,c,target_var))

plt.plot(error_array)
plt.show();