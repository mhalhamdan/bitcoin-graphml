import numpy as np
import pandas as pd
from numpy.random import permutation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

def upsample_data(X, y):

    df = pd.concat([X, y['class']], axis=1, join='inner')

    print(df.shape)

    df_majority = df[df['class'] == 0]
    df_minority = df[df['class'] == 1]

    print(df_majority.shape)

    print(df_minority.shape)

    print(df_majority.shape[0])

    # print(df_minority.head())
    # print(df_majority.head())

    df_minority_upsampled = resample(df_minority, 
                                     replace=True, 
                                     n_samples=df_majority.shape[0],
                                     random_state=123)

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    return df_upsampled.drop(['class'], axis=1), df_upsampled[['txId', 'class']]

def find_corr(xFeat, y, plot_title=False):

    new_df = pd.concat([xFeat, y], axis=1, join='inner')

    feature_matrix = xFeat.corr(method='pearson')
    label_matrix = new_df.corr(method='pearson')

    if plot_title:
        ax = sns.heatmap(label_matrix.abs(), xticklabels=True, yticklabels=True)
        plt.title(plot_title)
        # plt.show()

    return feature_matrix.abs(), label_matrix.abs()

def get_high_correlation(matrix, THRESHOLD=0.95):

    to_remove = []

    for idx, row in matrix.iterrows():
        # print(row[1])
        for col in row.index:
            # print(row[col])
            if row[col] > THRESHOLD and idx != col:
                if col not in to_remove:
                    to_remove.append(col)

    return to_remove

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


