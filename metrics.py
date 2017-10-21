import numpy as np
from numpy import genfromtxt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

P = np.zeros((4, 4))

def count_metrics():
    # with open('validation_labels.txt') as f:
    #     test_labels = f.readlines()
    with open('test_labels.txt') as f:
        test_labels = f.readlines()
    with open('predicted_labels.txt') as f:
        predicted_labels = f.readlines()
    for index in range(len(test_labels)):
        actual_label = int(eval(test_labels[index].strip()))
        predicted_label = int(eval(predicted_labels[index].strip()))
        P[actual_label][predicted_label] = P[actual_label][predicted_label] + 1

def F1n():
    return (2.0 * P[0][0]) / (np.sum(P[0]) + np.sum(P[:,0]))

def F1a():
    return (2.0 * P[1][1]) / (np.sum(P[1]) + np.sum(P[:,1]))

def F1o():
    return (2.0 * P[2][2]) / (np.sum(P[2]) + np.sum(P[:,2]))

def F1p():
    return (2.0 * P[3][3]) / (np.sum(P[3]) + np.sum(P[:,3]))

def F1():
    count_metrics()
    return (F1n() + F1a() + F1o()) / 3

def get_data():
    training_samples = genfromtxt('training_set.csv', delimiter=',', usecols = range(13))
    training_labels = genfromtxt('training_labels.txt', delimiter=',', usecols = range(1) , dtype=None)
    validation_samples = genfromtxt('validation_set.csv', delimiter=',', usecols = range(13))
    validation_labels = genfromtxt('validation_labels.txt', delimiter=',', usecols = range(1) , dtype=None)
    test_samples = genfromtxt('test_set.csv', delimiter=',', usecols = range(13))
    return training_samples, validation_samples, test_samples, training_labels, validation_labels

def apply_PCA(data):
    pca = PCA(n_components=5)
    return pca.fit_transform(data)

def feature_selection(data, labels):
    clf = ExtraTreesClassifier()
    clf = clf.fit(data, labels)
    print clf.feature_importances_ 
    model = SelectFromModel(clf, prefit=True)
    data = model.transform(data)
    return data

