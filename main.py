
print("Importing...")
from preprocessing import holdout
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

def predict(model, xTrain, yTrain, xTest, yTest):
    # Accuracy
    yHatTrain = model.predict(xTrain)
    trainAcc = metrics.accuracy_score(yTrain['class'], yHatTrain)

    yHatTest = model.predict(xTest)
    testAcc = metrics.accuracy_score(yTest['class'], yHatTest)

    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    yHatTest = model.predict_proba(xTest)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTrain['class'],
                                             yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest['class'],
                                             yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)

    return trainAcc, trainAuc, testAcc, testAuc

def train(xTrain, yTrain, model_name="knn", grid_search=False):
    print("Training for: ", model_name)
    model = None # Actual model to return

    # K-nearest neighbors
    if model_name == "knn":
        if grid_search:
            model = GridSearchCV(
                KNeighborsClassifier(), 
                [{'n_neighbors': range(1,10,2), 'metric': ['euclidean','manhattan']}], cv=5, scoring='f1_macro' , verbose=1)
        else:
            model = KNeighborsClassifier()

        model.fit(xTrain, yTrain['class'])
        
    # Decision tree
    elif model_name == "dt":
        if grid_search:
            parameters = {"max_depth":[1,3,5,7,9,11], 'min_samples_leaf':[50,100,150,200,250,300], "criterion":['gini', 'entropy']}
            model = GridSearchCV(
                DecisionTreeClassifier(), 
                parameters
            )
            print("DecisionTree best params: ", model.best_params_)
        else:
            model = DecisionTreeClassifier()

        model.fit(xTrain, yTrain['class'])


    # Gradient Descent Boosted Decision Tree (GDBDT)
    elif model_name == "GDBDT":
        pass

    # Graph Convulotional Network (GCN)
    elif model_name == "GCN":
        pass

    # Return trained model
    return model



def main():
    # Read data
    print("Reading data...")
    y = pd.read_csv("filtered_classes.csv")
    xFeat = pd.read_csv("filtered_features.csv")
    # Split data, train = 70%, test 30%
    xTrain, xTest, yTrain, yTest = holdout(xFeat, y, 0.7)

    # Initialize and train model
    model = train(xTrain, yTrain, "dt", grid_search=False)

    # Predict
    trainAcc, trainAuc, testAcc, testAuc = predict(model, xTrain, yTrain, xTest, yTest)
    print("Train accuracy: ", trainAcc)
    print("Test accuracy: ", testAcc)
    print("Train AUC: ", trainAuc)
    print("Test AUC: ", testAuc)


    



if __name__ == "__main__":
    main()
