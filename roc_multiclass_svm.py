import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


print"-----------------read in white wine dataset---------------------------"
white = pd.read_csv('E:/celine_sju/cus615/WineQualityData/winequality_white.csv', sep=(';'))
white['quality_class'] = white.quality.apply(lambda x: 0 if x <= 4 else 2 if x >= 7 else 1 )     
print white.head()
NColumns = 11
X0 = white.ix[:,0:NColumns].as_matrix()
y0 = white.quality_class.values

# Binarize the output
y = label_binarize(y0, classes=[0, 1, 2])
n_classes = y.shape[1]


# shuffle and split training and test sets
X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y, test_size=.3,
                                                    random_state=0)
# Normalizing wine dataset
from sklearn import preprocessing  
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X0_train)
X0_train = minmax_scale.transform(X0_train)
X0_test = minmax_scale.transform(X0_test)


# Learn to predict each class against the other
print"------------------------------ROC curve for SVM-----------------------"
clf1 = OneVsRestClassifier(svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma=1, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False))
y_score1 = clf1.fit(X0_train, y0_train).decision_function(X0_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y0_test[:, i], y_score1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y0_test.ravel(), y_score1.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#######################

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for multi-class classification-SVMs')
plt.legend(loc="lower right")
plt.show()

