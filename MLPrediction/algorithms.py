from random import seed
from numpy.random import seed
seed(1)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

def supportVector(x_train, x_test, y_train):

    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_train_SVM = clf.predict(x_train)
    y_test_SVM = clf.predict(x_test)
    return y_train_SVM, y_test_SVM

def kNeighbours(x_train, x_test, y_train):

    model = KNeighborsClassifier()
    model.fit(x_train,y_train)
    y_train_KNN = model.predict(x_train)
    y_test_KNN = model.predict(x_test)
    return y_train_KNN, y_test_KNN

def logisticRegression(x_train, x_test, y_train):
    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)
    y_train_LR = logreg.predict(x_train)
    y_test_LR = logreg.predict(x_test)

    return y_train_LR, y_test_LR

def decisionTree(x_train, x_test, y_train):
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train,y_train)
    y_train_DT = clf.predict(x_train)
    y_test_DT = clf.predict(x_test)

    return y_train_DT, y_test_DT


def randomForest(x_train, x_test, y_train):
    rclf = RandomForestClassifier()
    rclf.fit(x_train,y_train)
    y_train_RF = rclf.predict(x_train)
    y_test_RF = rclf.predict(x_test)

    return y_train_RF, y_test_RF

def preprocessingCancer(data):

    # Due to the data being loaded from the sklearn "toy datasets" there is no cleaning required and they've processed it already
    features = data.data
    labels = data.target

    # Produces a Train set of 75%, Test of 25% identical separation as article
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)
    return x_train, x_test, y_train, y_test

def preprocessingHeart(data):
    features = data.data
    labels = data.target

    # Produces a Train set of 75%, Test of 25% identical separation as article
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)
    return x_train, x_test, y_train, y_test

def evaluate(y_true, y_pred):
    acc_score = accuracy_score(y_true, y_pred)
    print(acc_score)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cnf_matrix.ravel()
    print("TN: ", tn)
    print("FP: ", fp)
    print("FN: ", fn)
    print("TP: ", tp)
    cls_report = classification_report(y_true, y_pred)
    print(cls_report)

def produceResults(x_train, x_test, y_train, y_test):
    # SVM
    trainPredSVM, testPredSVM = supportVector(x_train, x_test, y_train)
    print("##### SUPPORT VECTOR MACHINE #####")
    evaluate(y_train, trainPredSVM)
    evaluate(y_test, testPredSVM)

    # Random Forest
    trainPredRF, testPredRF = randomForest(x_train, x_test, y_train)
    print("##### RANDOM FOREST #####")
    evaluate(y_train, trainPredRF)
    evaluate(y_test, testPredRF)

    # Logistic Regression
    trainPredLR, testPredLR = logisticRegression(x_train, x_test, y_train)
    print("##### LOGISTIC REGRESSION #####")
    evaluate(y_train, trainPredLR)
    evaluate(y_test, testPredLR)

    # Decision Tree (C4.5)
    trainPredDT, testPredDT = decisionTree(x_train, x_test, y_train)
    print("##### DECISION TREE #####")
    evaluate(y_train, trainPredDT)
    evaluate(y_test, testPredDT)

    # K-Nearest Neighbours
    trainPredKNN, testPredKNN = kNeighbours(x_train, x_test, y_train)
    print("##### K NEAREST NEIGHBOURS #####")
    evaluate(y_train, trainPredKNN)
    evaluate(y_test, testPredKNN)

def main():
    
    cancer = datasets.load_breast_cancer()
    x_train, x_test, y_train, y_test = preprocessingCancer(cancer)
    produceResults(x_train, x_test, y_train, y_test)

    heart = pd.read_csv('heart.csv')


if __name__ == "__main__":
    main()


