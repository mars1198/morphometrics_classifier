import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

#

import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import collections
import itertools
import time
import os
import re


def plot_confusion_matrix(y_true, y_pred, matrix_title):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    plt.title(matrix_title, fontsize=12)
    plt.show()
    
    

#def plot_confusion_matrix(y_true, y_pred, matrix_title):
#    plt.figure(figsize=(20, 20), dpi=100)
#    cf_matrix = confusion_matrix(y_true, y_pred)
#    true_labels = np.unique(y_true)
#    pred_labels = np.unique(y_pred)
#    x_axis_labels = np.arange(len(true_labels))
#    y_axis_labels = np.arange(len(pred_labels))
#    sns.heatmap(cf_matrix, annot=True, cmap='Blues')
#    plt.title(matrix_title, fontsize=12)
#    plt.xticks(x_axis_labels, true_labels, rotation=90)
#    plt.yticks(y_axis_labels, pred_labels, rotation=0)
#    plt.ylabel('Predicted label', fontsize=10)
#    plt.xlabel('True label', fontsize=10)
#    plt.show()



def run_classifier(clfr, x_train_data, y_train_data, x_test_data, y_test_data, acc_str, matrix_header_str):
    start_time = time.time()
    clfr.fit(x_train_data, y_train_data)
    y_pred = clfr.predict(x_test_data)
    print("%f seconds" % (time.time() - start_time))

    # confusion matrix 
    print(acc_str.format(accuracy_score(y_test_data, y_pred) * 100))
    plot_confusion_matrix(y_test_data, y_pred, matrix_header_str)

#load data
bankdata = pd.read_csv("data/all2.csv")

#Extract features lables
features = bankdata.drop('Taxon', axis=1)
features = features.drop('Size', axis=1)
features = features.iloc[0:]
lables = bankdata['Taxon']

X_train, X_test, y_train, y_test = train_test_split(features, lables, test_size = 0.30, random_state=42, stratify = lables)

print('Support Vector Machine starting ...')
cl = LinearSVC()
run_classifier(cl, X_train, y_train, X_test, y_test, "SVM Accuracy: {0:0.1f}%", "SVM Confusion matrix")

#Extra Trees
print('Extra Trees Classifier starting ...')
cl = ExtraTreesClassifier(n_jobs=1,  n_estimators=10, criterion='gini', min_samples_split=2,
                           max_features=50, max_depth=None, min_samples_leaf=1)
run_classifier(cl, X_train, y_train, X_test, y_test, "ET Accuracy: {0:0.1f}%", "Extra Trees Confusion matrix")

# Random Forest
print('Random Forest Classifier starting ...')
cl = RandomForestClassifier(n_jobs=1, criterion='entropy', n_estimators=10, min_samples_split=2)
run_classifier(cl, X_train, y_train, X_test, y_test, "RF Accuracy: {0:0.1f}%", "Random Forest Confusion matrix")

#knn
print('K-Nearest Neighbours Classifier starting ...')
cl = KNeighborsClassifier(n_neighbors=1, n_jobs=4)
run_classifier(cl, X_train, y_train, X_test, y_test, "KNN Accuracy: {0:0.1f}%",
               "K-Nearest Neighbor Confusion matrix")

#MyLittlePony
print('Multi-layer Perceptron Classifier starting ...')
clf = MLPClassifier()
run_classifier(clf, X_train, y_train, X_test, y_test, "MLP Accuracy: {0:0.1f}%",
               "Multi-layer Perceptron Confusion matrix")


#Gaussian Naive Bayes Classifier
print('Gaussian Naive Bayes Classifier starting ...')
clf = GaussianNB()
run_classifier(clf, X_train, y_train, X_test, y_test, "GNB Accuracy: {0:0.1f}%",
               "Gaussian Naive Bayes Confusion matrix")

#LDA
print('Linear Discriminant Analysis Classifier starting ...')
clf = LinearDiscriminantAnalysis()
run_classifier(clf, X_train, y_train, X_test, y_test, "LDA Accuracy: {0:0.1f}%",
               "Linear Discriminant Analysis Confusion matrix")

#QDA
print('Quadratic Discriminant Analysis Classifier starting ...')
clf = QuadraticDiscriminantAnalysis()
run_classifier(clf, X_train, y_train, X_test, y_test, "QDA Accuracy: {0:0.1f}%",
               "Quadratic Discriminant Analysis Confusion matrix")
