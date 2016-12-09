import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report

#read in white wine dataset 
print"-----------------read in white wine dataset---------------------------"
white = pd.read_csv('E:/celine_sju/cus615/WineQualityData/winequality_white.csv', sep=(';'))
white['quality_class'] = white.quality.apply(lambda x: 0 if x <= 4 else 2 if x >= 7 else 1 )     
print white.head()
X = white.ix[:,0:11].as_matrix()
y = white.quality_class.values

# split dataset into training and testing dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)


# Normalizing wine dataset
from sklearn import preprocessing  
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X_train)
X_train = minmax_scale.transform(X_train)
X_test = minmax_scale.transform(X_test)

target_names = ['class 0', 'class 1', 'class 2']

###############################################################################
#1.run KNN classifier
print "----------------------run KNN classifier-------------------------------"
from sklearn.neighbors import KNeighborsClassifier as KNN

knn_clf = KNN(n_neighbors=33, weights='distance')
pred_test_knn = knn_clf.fit(X_train, y_train).predict(X_test)
    
print('KNN clf Accurancy:')
print('{:.2%}'.format(metrics.accuracy_score(y_test, pred_test_knn)))
    
print('KNN clf MSE:') 
print('{:.2}'.format(metrics.mean_squared_error(y_test, pred_test_knn)))
    
print('Classification Report of the KNN:')
print(classification_report(y_test, pred_test_knn, target_names=target_names))

    
# Finding optimal K
results = []    
for n in range(1, 50, 2):    
    clf = KNN(n_neighbors=n,weights='distance' )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    accuracy = np.where(preds==y_test, 1, 0).sum() / float(len(y_test))
    print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)
    results.append([n, accuracy])

results = pd.DataFrame(results, columns=["n", "accuracy"])

plt.plot(results.n, results.accuracy)
plt.title("Accuracy with Increasing K")
plt.show()

###############################################################################

# 2.RandomForest Classifier
print"------------------RandomForest Classifier------------------------------"
from sklearn.ensemble import RandomForestClassifier
rand_clf = RandomForestClassifier(n_estimators= 15)
pred_test_rand = rand_clf.fit(X_train, y_train).predict(X_test)

print('RandomForest clf Accurancy:')
print('{:.2%}'.format(metrics.accuracy_score(y_test, pred_test_rand)))

print('RandomForest clf MSE:')
print('{:.2}'.format(metrics.mean_squared_error(y_test, pred_test_rand))) 

print('Classification Report of the RandomForest-classifier:')
print(classification_report(y_test, pred_test_rand, target_names=target_names))



###############################################################################
#3.Run SVM classifier
# Specify grid parameters for the support vector machines classifier (SVC)
print"---------------------------SVM classifier--------------------------------"
from sklearn.svm import SVC

svm_classifier =SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma=1, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
pred_test_svm = svm_classifier.fit(X_train, y_train).predict(X_test)
 

print('SVM clf Accurancy:')
print('{:.2%}'.format(metrics.accuracy_score(y_test, pred_test_svm)))

print('SVM clf MSE:')
print('{:.2}'.format(metrics.mean_squared_error(y_test, pred_test_svm)))

print('Classification Report of the SVM:')
print(classification_report(y_test,pred_test_svm, target_names=target_names))


###############################################################################
## 4. Run Arificial Netual classifier
print"-------------------Arificial Netual classifier--------------------------"

from sklearn.neural_network import MLPClassifier
ann_classifier = MLPClassifier(solver='lbfgs', alpha=1, learning_rate='adaptive',hidden_layer_sizes=(5, 2), random_state=1)
pred_test_ann = ann_classifier.fit(X_train, y_train).predict(X_test)

print('ANN clf Accurancy:')
print('{:.2%}'.format(metrics.accuracy_score(y_test, pred_test_ann)))

print('ANN clf MSE:')
print('{:.2}'.format(metrics.mean_squared_error(y_test, pred_test_ann))) 

print('Classification Report of the ANN:')
print(classification_report(y_test,pred_test_ann, target_names=target_names))


###############################################################################
# ploting confusion matrix
import itertools
from sklearn.metrics import confusion_matrix

print"--------------------ploting confusion matrix----------------------------"
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.around(cm,decimals=2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Plot non-normalized confusion matrix-knn
cnf_matrix_knn = confusion_matrix(y_test, pred_test_knn)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix_knn, classes=target_names,
                      title='Confusion matrix-KNN')
plt.show()

# Plot normalized confusion matrix-knn
plt.figure()
plot_confusion_matrix(cnf_matrix_knn, classes=target_names, normalize=True,
                      title='Normalized confusion matrix-KNN')

plt.show()

# Plot non-normalized confusion matrix-RandomForest                 
cnf_matrix_rand = confusion_matrix(y_test, pred_test_rand)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix_rand, classes=target_names,
                      title='Confusion matrix-RandomForest')
plt.show()

# Plot normalized confusion matrix-RandomForest
plt.figure()
plot_confusion_matrix(cnf_matrix_rand, classes=target_names, normalize=True,
                      title='Normalized confusion matrix-RandomForest')

plt.show()

# Plot non-normalized confusion matrix-svm
cnf_matrix_svm = confusion_matrix(y_test, pred_test_svm)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix_svm, classes=target_names,
                      title='Confusion matrix-SVM')
plt.show()

# Plot normalized confusion matrix-SVM
plt.figure()
plot_confusion_matrix(cnf_matrix_svm, classes=target_names, normalize=True,
                      title='Normalized confusion matrix-SVM')

plt.show()

# Plot non-normalized confusion matrix-ANN
cnf_matrix_ann = confusion_matrix(y_test, pred_test_ann)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix_ann, classes=target_names,
                      title='Confusion matrix-ANN')
plt.show()

# Plot normalized confusion matrix-ANN
plt.figure()
plot_confusion_matrix(cnf_matrix_ann, classes=target_names, normalize=True,
                      title='Normalized confusion matrix-ANN')

plt.show()