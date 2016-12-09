import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math 
import random
from numpy.random import permutation
from sklearn.lda import LDA 
from sklearn.linear_model import LogisticRegression as LogR
from scipy import interp  
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc 

#-- user adjustable variables
test_sample_percent = 0.3

#read in red wine dataset and white wine dataset and combine the two datasets
r_wine = pd.read_csv('C:/Users/Aden/Dropbox/cf/CUS615/WineQualityData/winequality_red.csv', sep=(';'))
w_wine = pd.read_csv('C:/Users/Aden/Dropbox/cf/CUS615/WineQualityData/winequality_white.csv', sep=(';'))


print "---- starting training :test_sample_percent", test_sample_percent, "-----------------"
r_wine['type'] = 1
w_wine['type'] = 0
wines = r_wine.append(w_wine, ignore_index=True)
print wines.head()
X=wines.ix[:,0:11]
y=wines.type
y = np.ravel(y)


# split dataset into training and testing dataset
# Randomly shuffle the index of nba.
random_indices = permutation(wines.index)
# Set a cutoff for how many items we want in the test set(in this case 1/3 of the items)
test_cutoff = int(math.floor(len(wines)*test_sample_percent))
test = wines.loc[random_indices[1:test_cutoff]]
train = wines.loc[random_indices[test_cutoff:]]
X_train = train.ix[:,0:11]
y_train = train.type
X_test = test.ix[:,0:11]
y_test = test.type


# Normalizing wine dataset
from sklearn import preprocessing  
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train = std_scale.transform(X_train)
X_test = std_scale.transform(X_test)

target_names = ['class 0', 'class 1']

#############################################################################

print "---- starting LDA training -----------------"
# LDA classifier
from sklearn.metrics import classification_report

lda_clf = LDA(n_components=None, priors=None)
lda_clf.fit(X_train, y_train)
# prediction
pred_test_lda = lda_clf.predict(X_test)
df_lda = pd.DataFrame({'Actual':y_test,'Prediction':pred_test_lda})
print df_lda.head()

print('Accurancy:')
print('{:.2%}'.format(metrics.accuracy_score(y_test, pred_test_lda)))

print('MSE:')
print('{:.2}'.format(metrics.mean_squared_error(y_test, pred_test_lda)))

print('Confusion Matrix of the LDA-classifier:')
print(metrics.confusion_matrix(y_test, lda_clf.predict(X_test)))

print('Classification Report of the ANN:')
print(classification_report(y_test,pred_test_lda, target_names=target_names))


##############################################################################

print "---- starting Logistic training -----------------"
# Logistic Regression

logR_clf = LogR(random_state=123456789)
logR_clf.fit(X_train, y_train)
pred_test_logR = logR_clf.predict(X_test)

df_logR = pd.DataFrame({'Actual':y_test,'Prediction':pred_test_logR})
print df_logR.head()

print('Accurancy:')
print('{:.2%}'.format(metrics.accuracy_score(y_test, pred_test_logR)))

print('MSE:')
print('{:.2}'.format(metrics.mean_squared_error(y_test, pred_test_logR)))

print('Confusion Matrix of the LDA-classifier:')
print(metrics.confusion_matrix(y_test, logR_clf.predict(X_test)))
###############################################################################