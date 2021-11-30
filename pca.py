import pandas as pd
import numpy as np
from preprocessing import get_high_correlation, holdout, find_corr, get_high_correlation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def normalize_data(xTrain, xTest):
    stdScale = StandardScaler()
    stdScale.fit(xTrain)
    trainScaled = stdScale.transform(xTrain)
    testScaled = stdScale.transform(xTest)
    return trainScaled, testScaled

def apply_PCA(n_components, xFeat):
    b_pca = PCA(n_components=n_components)
    b_pca.fit_transform(xFeat)
    return b_pca

def sum_variances(variances):
    sum = 0
    for variance in variances:
        sum = sum + variance
    return sum

def main():
    y = pd.read_csv("filtered_classes.csv")
    xFeat = pd.read_csv("filtered_features.csv")
    xTrain, xTest, yTrain, yTest = holdout(xFeat, y, 0.7)

    xTrain, xTest = normalize_data(xTrain, xTest)

    # iterate through various n components
    n_components = [167, 150, 125, 100, 75, 50, 25, 15, 10, 5, 1]
    for n_component in n_components:
        n_PCA = apply_PCA(n_components=n_component, xFeat=xTrain).explained_variance_ratio_
        n_sum = sum_variances(n_PCA)
        print("  N components: ", n_component, " Sum: ", n_sum)

if __name__ == "__main__":
    main()