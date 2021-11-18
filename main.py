from matplotlib.pyplot import axis
from sklearn import model_selection
from sklearn import neighbors
from preprocessing import get_high_correlation, holdout, find_corr, get_high_correlation
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics


def predict(model, xTrain, yTrain, xTest, yTest):
    # Test
    # Accuracy
    yHatTest = model.predict(xTest)
    testAcc = metrics.accuracy_score(yTest['class'], yHatTest)

    # predict training and testing probabilties
    yHatTest = model.predict_proba(xTest)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest['class'],
                                            yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)

    test_precision = metrics.precision_score(yTest['class'], yHatTest[:,1].astype(int)) 
    test_recall = metrics.recall_score(yTest['class'], yHatTest[:,1].astype(int))
    test_matrix = metrics.confusion_matrix(yTest['class'], yHatTest[:,1].astype(int)).ravel()

    # Train
    # Accuracy
    yHatTrain = model.predict(xTrain)
    trainAcc = metrics.accuracy_score(yTrain['class'], yHatTrain)

    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTrain['class'],
                                            yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)

    train_precision = metrics.precision_score(yTrain['class'], yHatTrain[:,1].astype(int))
    train_recall = metrics.recall_score(yTrain['class'], yHatTrain[:,1].astype(int))
    train_matrix = metrics.confusion_matrix(yTrain['class'], yHatTrain[:,1].astype(int)).ravel()

    result_metrics = {
        "trainAcc": trainAcc, 
        "trainAuc": trainAuc, 
        "testAcc": testAcc, 
        "testAuc": testAuc,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_(tn, fp, fn, tp)": train_matrix,
        "test_(tn, fp, fn, tp)": test_matrix
        }

    return result_metrics


def train(xTrain, yTrain, model_name="knn", grid_search=False):
    model = None # Actual model to return

    # K-nearest neighbors
    if model_name == "knn":
        if grid_search:
            model = GridSearchCV(
                KNeighborsClassifier(), 
                [{'n_neighbors': range(1,10,2), 'metric': ['euclidean','manhattan']}], cv=5, scoring='f1_macro' , verbose=1)
        else:
            model = KNeighborsClassifier(metric="manhattan")

        model.fit(xTrain, yTrain['class'])
        
    # Decision tree
    elif model_name == "dt":
        if grid_search:
            parameters = {"max_depth":[1,3,5,7,9,11], 'min_samples_leaf':[2,10,50,100,150,200,250,300], "criterion":['gini', 'entropy']}
            model = GridSearchCV(
                DecisionTreeClassifier(), 
                parameters
            )
            
        else:
            # {'criterion': 'entropy', 'max_depth': 11, 'min_samples_leaf': 50}
            model = DecisionTreeClassifier(criterion='entropy', max_depth=11, min_samples_leaf=10)

        model.fit(xTrain, yTrain['class'])
        print("DecisionTree best params: ", model.best_params_)

    # Gradient Descent Boosted Decision Tree (GDBDT)
    elif model_name == "GDBDT":
        pass

    # Graph Convulotional Network (GCN)
    elif model_name == "GCN":
        pass

    # Return trained model
    return model


def main():
    model_name = "dt"
    # Read data
    print("Reading data...")
    y = pd.read_csv("filtered_classes.csv")
    xFeat = pd.read_csv("filtered_features.csv")

    # Find correlation matrix
    feature_matrix, label_matrix = find_corr(xFeat, y, plot_title=False)
    
    # Threshold for cutting out a correlation 
    to_remove = get_high_correlation(feature_matrix,THRESHOLD=0.95)

    # Drop based on correlation
    xFeat.drop(to_remove, axis=1, inplace=True)

    # Get correlation after dropping
    _, _ = find_corr(xFeat, y.drop(["txId"], axis=1), plot_title="Correlation Matrix for Bitcoin dataset")

    # Split data, train = 70%, test 30%
    xTrain, xTest, yTrain, yTest = holdout(xFeat, y, 0.7)

    model_name = "knn"

    # Initialize and train model
    model = train(xTrain, yTrain, model_name, grid_search=False)

    try:
        print(f"{model_name} best params: {model.best_params_}")
    except:
        pass

    print("Predicting...")
    results = predict(model, xTrain, yTrain, xTest, yTest)

    for key, value in results.items():
        print(f"{key} : {value}")


if __name__ == "__main__":
    main()

# testing github
