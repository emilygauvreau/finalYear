'''
Course Project - Wisconsin Breast Cancer Diagnostic 
Emily Gauvreau
20074874 - 17emg 

This program implements 5 different machine learning algorithms from the library sklearn
Each one is contained in a separate function which created a "y_pred" variable
These predictions are then feed into the evaluation function to produce performance metrics
'''

# uncomment to ensure randomization is consistent
# from random import seed
# seed(10)
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

def supportVector(x_train, x_test, y_train):
    """ 
    This function is responsible for implementing the Support Vector Machine Algorithm
    Parameters:
        - x_train: the feature values for the training set
        - x_test: the feature values for the test set
        - y_train: the labels for the training set
    Returns:
        - y_train_SVM: the predictions for the training set
        - y_test_SVM: the predictions for the test set
    """

    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_train_SVM = clf.predict(x_train)
    y_test_SVM = clf.predict(x_test)
    return y_train_SVM, y_test_SVM

def kNeighbours(x_train, x_test, y_train):
    """ 
    This function is responsible for implementing the K-Nearest Neighbors algorithm
    Parameters:
        - x_train: the feature values for the training set
        - x_test: the feature values for the test set
        - y_train: the labels for the training set
    Returns:
        - y_train_KNN: the predictions for the training set
        - y_test_KNN: the predictions for the test set
    """
    model = KNeighborsClassifier()
    model.fit(x_train,y_train)
    y_train_KNN = model.predict(x_train)
    y_test_KNN = model.predict(x_test)
    return y_train_KNN, y_test_KNN

def logisticRegression(x_train, x_test, y_train):
    """ 
    This function is responsible for implementing the Logistic Regression algorithm
    Parameters:
        - x_train: the feature values for the training set
        - x_test: the feature values for the test set
        - y_train: the labels for the training set
    Returns:
        - y_train_LR: the predictions for the training set
        - y_test_LR: the predictions for the test set
    """
    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)
    y_train_LR = logreg.predict(x_train)
    y_test_LR = logreg.predict(x_test)

    return y_train_LR, y_test_LR

def decisionTree(x_train, x_test, y_train):
    """ 
    This function is responsible for implementing the Decision Tree algorithm
    Parameters:
        - x_train: the feature values for the training set
        - x_test: the feature values for the test set
        - y_train: the labels for the training set
    Returns:
        - y_train_DT: the predictions for the training set
        - y_test_DT: the predictions for the test set
    """
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train,y_train)
    y_train_DT = clf.predict(x_train)
    y_test_DT = clf.predict(x_test)

    return y_train_DT, y_test_DT


def randomForest(x_train, x_test, y_train):
    """ 
    This function is responsible for implementing the Random Forest algorithm
    Parameters:
        - x_train: the feature values for the training set
        - x_test: the feature values for the test set
        - y_train: the labels for the training set
    Returns:
        - y_train_RF: the predictions for the training set
        - y_test_RF: the predictions for the test set
    """
    rclf = RandomForestClassifier()
    rclf.fit(x_train,y_train)
    y_train_RF = rclf.predict(x_train)
    y_test_RF = rclf.predict(x_test)

    return y_train_RF, y_test_RF

def preprocessingCancer(data):
    """ 
    This function prepares the data for the prediction algorithms and splis it into a test and training set
    Parameters:
        - data: the dataset that contains features and labels
    Returns:
        - x_train: the feature values for the training set
        - x_test: the feature values for the test set
        - y_train: the labels for the training set
        - y_test: the labels for the test set
    """

    # Due to the data being loaded from the sklearn "toy datasets" there is no cleaning required and they've processed it already
    features = data.data
    labels = data.target

    # Produces a Train set of 75%, Test of 25% identical separation as article
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)
    return x_train, x_test, y_train, y_test

def preprocessingHeart(data):
    """ 
    This function prepares the data for the prediction algorithms and splis it into a test and training set
    Parameters:
        - data: the dataset that contains features and labels
    Returns:
        - x_train: the feature values for the training set
        - x_test: the feature values for the test set
        - y_train: the labels for the training set
        - y_test: the labels for the test set
    """
    # Both return all zeros which means there are no missing values that need replacing
    # print("NaN Values: ", data.isna().sum())
    # print("Null Values: ", data.isnull().sum())

    # Must replace the categorical variables
    # Male = 1 Female = 0  AnginaNo = 0 AnginaYes = 1
    single = LabelEncoder()
    data['Sex'] = single.fit_transform(data['Sex'])
    data['ExerciseAngina'] = single.fit_transform(data['ExerciseAngina'])
    data = pd.get_dummies(data)

    features = data.drop('HeartDisease', axis=1)
    labels = data['HeartDisease']

    # Produces a Train set of 75%, Test of 25% identical separation as article
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)
    
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    return x_train, x_test, y_train, y_test

def evaluate(y_true, y_pred):
    """
    This function takes the predicted labels and the actual labels and computes the performance metrics accordingly
    Parameters:
        - y_true: the labels that are pre-assigned to the data
        - y_pred: the labels that are predicted for the data
    """
    acc_score = round(accuracy_score(y_true, y_pred) * 100, 2)
    print(acc_score)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cnf_matrix.ravel()
    print("TN: ", tn, " FP: ", fp, " FN: ", fn, " TP: ", tp)
    cls_report = classification_report(y_true, y_pred)
    print(cls_report)

# This function runs all of the algorithms and evaluation functions
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
    x_trainC, x_testC, y_trainC, y_testC = preprocessingCancer(cancer)
    produceResults(x_trainC, x_testC, y_trainC, y_testC)

    heart = pd.read_csv('heart.csv')
    x_trainH, x_testH, y_trainH, y_testH = preprocessingHeart(heart)
    produceResults(x_trainH, x_testH, y_trainH, y_testH)



if __name__ == "__main__":
    main()


