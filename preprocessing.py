import numpy as np
import pandas as pd
from numpy.random import permutation

def prepare_date():
    # Create headers for features file
    features_header = ["txId", "timestep"]
    for i in range(1, 166): features_header.append(f"feature{i}")

    # Read files
    classes = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_classes.csv")
    features = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_features.csv", names=features_header)
    edges = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")

    # Extract only instances with a known label 
    filtered_classes = classes[classes['class'] != "unknown"]
    # filtered_classes = classes[classes['class'] == "2" and classes['class'] == "1"]
    filtered_features = features.loc[filtered_classes.index]

    # Case: Make labels binary
    filtered_classes.replace("2", "0", inplace=True)

    # Save new files
    filtered_classes.to_csv("filtered_classes.csv", index=False)
    filtered_features.to_csv("filtered_features.csv", index=False)

def holdout(xFeat, y, testSize):
    """
    Split xFeat into random train and test based on the testSize and
    return the model performance on the training and test set. 
    """
    # Shuffle indexes
    p = permutation(len(xFeat))
    xFeat = xFeat.loc[p]
    y = y.loc[p]

    # Second: find split_index from testSize
    split_index = round(testSize*len(y))

    # Split data
    xTrain = xFeat.iloc[0:split_index].reset_index()
    xTest = xFeat.iloc[split_index:len(y)].reset_index()

    yTrain = y.iloc[0:split_index].reset_index()
    yTest = y.iloc[split_index:len(y)].reset_index()

    # Drop index
    xTrain = xTrain.drop(columns="index")
    yTrain = yTrain.drop(columns="index")
    xTest = xTest.drop(columns="index")
    yTest = yTest.drop(columns="index")

    # Compute
    return xTrain, xTest, yTrain, yTest

if __name__ == "__main__":
    # Examples:

    # First step
    prepare_date()

    # Second step
    y = pd.read_csv("filtered_classes.csv")
    xFeat = pd.read_csv("filtered_features.csv")

    xTrain, xTest, yTrain, yTest = holdout(xFeat, y, 0.7)

    print(xTrain.head())
    print(yTrain.head())
    print(xTest.head())
    print(yTest.head())


