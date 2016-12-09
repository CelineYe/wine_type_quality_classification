# data pre-processing and exploring
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 20)
pd.set_option('precision', 3)

#read in red wine dataset and white wine dataset
r_wine = pd.read_csv('E:/celine_sju/cus615/WineQualityData/winequality_red.csv', sep=(';'))
w_wine = pd.read_csv('E:/celine_sju/cus615/WineQualityData/winequality_white.csv', sep=(';'))
print r_wine.shape
print r_wine.head(3)
print w_wine.shape
print w_wine.head(3)

# count N/A values
print r_wine.isnull().sum()
print w_wine.isnull().sum()

# summary of the dataset
print r_wine.describe()
print w_wine.describe()

# plot histgrams for each variable. 
plt.style.use('ggplot')
pd.DataFrame.hist(r_wine, figsize = (15,15)) 
pd.DataFrame.hist(w_wine, figsize = (15,15)) 


# correlaton of variables
print r_wine.corr(method='pearson')
print w_wine.corr(method='pearson')

import seaborn as sns
sns.pairplot(r_wine[["fixed acidity","volatile acidity","citric acid","residual sugar",
"chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]])
sns.pairplot(w_wine[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
"free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]])


#Rad visualization
from pandas.tools.plotting import radviz
plt.figure()
radviz(r_wine, 'quality')
plt.figure()
radviz(w_wine, 'quality')

#parallel_coordinates visualization
from pandas.tools.plotting import parallel_coordinates
plt.figure(figsize=(20, 20), dpi=80)
parallel_coordinates(r_wine, 'quality', colormap='gist_rainbow')
plt.figure(figsize=(20, 20), dpi=80)
parallel_coordinates(w_wine, 'quality', colormap='gist_rainbow')