from matplotlib.pyplot import axis, title
from sklearn import model_selection
from sklearn import neighbors
from pca import apply_PCA, normalize_data
from preprocessing import get_high_correlation, holdout, find_corr, get_high_correlation, upsample_SMOTE, upsample_data, minority_only
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics



# metrics.
import matplotlib.pyplot as plt


def predict(model, xTrain, yTrain, xTest, yTest):
    # Test
    # Accuracy
    yHatTest = model.predict(xTest)
    testAcc = metrics.accuracy_score(yTest, yHatTest)

    # predict training and testing probabilties
    yHatTest = model.predict_proba(xTest)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest,
                                            yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)

    test_precision = metrics.precision_score(yTest, yHatTest[:,1].astype(int)) 
    test_recall = metrics.recall_score(yTest, yHatTest[:,1].astype(int))
    test_matrix = metrics.confusion_matrix(yTest, yHatTest[:,1].astype(int)).ravel()

    # Train
    # Accuracy
    yHatTrain = model.predict(xTrain)
    trainAcc = metrics.accuracy_score(yTrain, yHatTrain)

    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTrain,
                                            yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)

    train_precision = metrics.precision_score(yTrain, yHatTrain[:,1].astype(int))
    train_recall = metrics.recall_score(yTrain, yHatTrain[:,1].astype(int))
    train_matrix = metrics.confusion_matrix(yTrain, yHatTrain[:,1].astype(int)).ravel()

    f1_score = metrics.f1_score(yTest, yHatTest[:,1].astype(int))

    result_metrics = {
        "train_acc": trainAcc, 
        "train_auc": trainAuc, 
        "test_acc": testAcc, 
        "test_auc": testAuc,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_(tn, fp, fn, tp)": train_matrix,
        "test_(tn, fp, fn, tp)": test_matrix,
        "f1_score": f1_score
        }

    return result_metrics


def train(xTrain, yTrain, model_name="knn", grid_search=False):
    model = None # Actual model to return

    # K-nearest neighbors
    if model_name == "knn":
        if grid_search:
            model = GridSearchCV(
                KNeighborsClassifier(), 
                [{'n_neighbors': [3, 5, 10], 'metric': ['euclidean','manhattan'], 'n_jobs': [4]}], cv=2, scoring='f1_macro' , verbose=3, n_jobs=4)
        else:
            # model = KNeighborsClassifier(metric="manhattan")
            model = KNeighborsClassifier(metric='manhattan', n_neighbors=3, n_jobs=8)

        model.fit(xTrain, yTrain)
        
    # Decision tree
    elif model_name == "dt":
        if grid_search:
            parameters = {"max_depth":[1,3,5,7,9,11], 'min_samples_leaf':[2,10,50,100], "criterion":['gini', 'entropy']}
            model = GridSearchCV(
                DecisionTreeClassifier(), 
                parameters,
                cv=2,
                n_jobs=4,
                verbose=3
            )
            
        else:
            # {'criterion': 'entropy', 'max_depth': 11, 'min_samples_leaf': 50}
            model = DecisionTreeClassifier(criterion='entropy', max_depth=11, min_samples_leaf=10)

        model.fit(xTrain, yTrain)

    # Gradient Descent Boosted Decision Tree (GDBDT)
    elif model_name == "GDBDT":
        if grid_search:
            parameters = {"learning_rate": [0.001, 0.00001, 0.000001], "max_depth": [5, 20], 'max_features': [2, 3], 'n_estimators': [100, 200]}
            model = GridSearchCV(
                GradientBoostingClassifier(), 
                parameters,
                cv=2,
                verbose=3,
                n_jobs=8
            )
            
        else:
            model = GradientBoostingClassifier()

        model.fit(xTrain, yTrain)

    # Random Forest (rf)
    elif model_name == "rf":
        if grid_search:
            # parameters = {'criterion': ['entropy', 'gini'], 'max_depth': [None, 5, 10]}
            parameters = {"n_estimators": [5,50,100, 200], "criterion": ["gini", "entropy"], "max_depth":[10,20,50], "min_samples_leaf":[1, 2, 10], "max_features": ["auto"]}
            model = GridSearchCV(
                RandomForestClassifier(),
                parameters, 
                verbose=3, 
                cv=2,
                n_jobs=4
            )

        else:
            # model = RandomForestClassifier(criterion='entropy', max_depth=100, min_samples_leaf=5, n_estimators=50)
            # model = RandomForestClassifier(criterion='gini', max_depth=20, min_samples_leaf=1)
            model = RandomForestClassifier(n_estimators=50, max_features=50, n_jobs=4)

        model.fit(xTrain, yTrain)

    # Graph Convulotional Network (GCN)
    elif model_name == "GCN":
        pass

    # Return trained model
    return model

def run_kfold(model_name, CORRELATION, PCA, UPSAMPLE, title):

    print("Model: ", model_name)
    # Read data
    print("Reading data...")
    y = pd.read_csv("filtered_classes.csv")
    xFeat = pd.read_csv("filtered_features.csv")
    print("Done!")

    if CORRELATION:
        # Find correlation matrix
        # feature_matrix, label_matrix = find_corr(xFeat, y, plot_title="Correlation Matrix for Bitcoin dataset")
        feature_matrix, label_matrix = find_corr(xFeat, y, plot_title=False)
        
        # Threshold for cutting out a correlation 
        to_remove = get_high_correlation(feature_matrix,THRESHOLD=0.95)

        # Drop based on correlation
        xFeat.drop(to_remove, axis=1, inplace=True)

        # Get correlation after dropping
        # _, _ = find_corr(xFeat, y.drop(["txId"], axis=1), plot_title="Correlation Matrix for Bitcoin dataset (after slicing)")
    
    # Split data, train = 70%, test 30%
    # xTrain, xTest, yTrain, yTest = holdout(xFeat, y, 0.7)

    kf = KFold(n_splits=3, random_state=None, shuffle=True)

    f1_score = []
    acc = []

    for train_index, test_index in kf.split(xFeat):
        print("TRAIN:", train_index, "TEST:", test_index)
        xTrain, xTest = xFeat.loc[train_index], xFeat.loc[test_index]
        yTrain, yTest = y.loc[train_index], y.loc[test_index]

        if UPSAMPLE == "+U":
            # Upsample data
            xTrain, yTrain = upsample_data(xTrain, yTrain)

        if UPSAMPLE == "SMOTE":
            xTrain, yTrain = upsample_SMOTE(xTrain, yTrain)

        # Drop txId
        if True:
            xTrain = xTrain.drop(['txId'], axis=1)
            xTest = xTest.drop(['txId'], axis=1)

        if PCA:
            # PCA
            xTrain, xTest = normalize_data(xTrain, xTest)
            pca = apply_PCA(58, xTrain, yTrain)
            xTrain = pca.transform(xTrain)
            xTest = pca.transform(xTest)
            print(xTrain.shape)

        
        print("Training...")
        # Initialize and train model
        model = train(xTrain, yTrain['class'], model_name, grid_search=False)
        print("Done!")

        try:
            print(f"{model_name} best params: {model.best_params_}")
        except:
            pass

        print("Predicting...")
        results = predict(model, xTrain, yTrain['class'], xTest, yTest['class'])
        print("Done!")

        for key, value in results.items():
            print(f"{key} : {value}")

        # ROC curve
        # metrics.plot_roc_curve(model, xTest, yTest['class'])
        # plt.title(f"{model_name} {title} ROC curve")
        # plt.savefig(f"{model_name} {title} ROC curve")

        # metrics.plot_confusion_matrix(model, xTest, yTest['class'])
        # plt.title(f"{model_name} {title} confusion matrix")
        # plt.savefig(f"{model_name} {title} confusion matrix")


        f1_score.append(results['f1_score'])
        acc.append(results['test_acc'])

    return {"test_acc": sum(acc)/len(acc), "f1_score": sum(f1_score)/len(f1_score)}

def run(model_name, CORRELATION, PCA, UPSAMPLE, title):

    print("Model: ", model_name)
    # Read data
    print("Reading data...")
    y = pd.read_csv("filtered_classes.csv")
    xFeat = pd.read_csv("filtered_features.csv")
    print("Done!")

    if CORRELATION:
        # Find correlation matrix
        # feature_matrix, label_matrix = find_corr(xFeat, y, plot_title="Correlation Matrix for Bitcoin dataset")
        feature_matrix, label_matrix = find_corr(xFeat, y, plot_title=False)
        
        # Threshold for cutting out a correlation 
        to_remove = get_high_correlation(feature_matrix,THRESHOLD=0.95)

        # Drop based on correlation
        xFeat.drop(to_remove, axis=1, inplace=True)

        # Get correlation after dropping
        # _, _ = find_corr(xFeat, y.drop(["txId"], axis=1), plot_title="Correlation Matrix for Bitcoin dataset (after slicing)")
    
    # Split data, train = 70%, test 30%
    xTrain, xTest, yTrain, yTest = holdout(xFeat, y, 0.7)

    if UPSAMPLE == "+U":
        # Upsample data
        xTrain, yTrain = upsample_data(xTrain, yTrain)

    if UPSAMPLE == "SMOTE":
        xTrain, yTrain = upsample_SMOTE(xTrain, yTrain)

    # Drop txId
    if True:
        xTrain = xTrain.drop(['txId'], axis=1)
        xTest = xTest.drop(['txId'], axis=1)

    if PCA:
        # PCA
        xTrain, xTest = normalize_data(xTrain, xTest)
        pca = apply_PCA(58, xTrain, yTrain)
        xTrain = pca.transform(xTrain)
        xTest = pca.transform(xTest)
        print(xTrain.shape)

    
    print("Training...")
    # Initialize and train model
    model = train(xTrain, yTrain['class'], model_name, grid_search=False)
    print("Done!")

    try:
        print(f"{model_name} best params: {model.best_params_}")
    except:
        pass

    print("Predicting...")
    results = predict(model, xTrain, yTrain['class'], xTest, yTest['class'])
    print("Done!")

    for key, value in results.items():
        print(f"{key} : {value}")

    # ROC curve
    metrics.plot_roc_curve(model, xTest, yTest['class'])
    plt.title(f"{model_name} {title} ROC curve")
    plt.savefig(f"{model_name} {title} ROC curve")

    metrics.plot_confusion_matrix(model, xTest, yTest['class'])
    plt.title(f"{model_name} {title} confusion matrix")
    plt.savefig(f"{model_name} {title} confusion matrix")

    return results

def main():
    model_name = "knn"
    print("Model: ", model_name)
    # Read data
    print("Reading data...")
    y = pd.read_csv("filtered_classes.csv")
    xFeat = pd.read_csv("filtered_features.csv")
    print("Done!")

    CORRELATION = False
    if CORRELATION:
        # Find correlation matrix
        # feature_matrix, label_matrix = find_corr(xFeat, y, plot_title="Correlation Matrix for Bitcoin dataset")
        feature_matrix, label_matrix = find_corr(xFeat, y, plot_title=False)
        
        # Threshold for cutting out a correlation 
        to_remove = get_high_correlation(feature_matrix,THRESHOLD=0.95)

        # Drop based on correlation
        xFeat.drop(to_remove, axis=1, inplace=True)

        # Get correlation after dropping
        # _, _ = find_corr(xFeat, y.drop(["txId"], axis=1), plot_title="Correlation Matrix for Bitcoin dataset (after slicing)")
        print(xFeat.shape)

    
    # Split data, train = 70%, test 30%
    xTrain, xTest, yTrain, yTest = holdout(xFeat, y, 0.7)

    # Testing reduced size
    if False:

        xTrain, xTest, yTrain, yTest = holdout(xFeat, y, 0.5)

        xTrain, xTest, yTrain, yTest = holdout(xTrain, yTrain, 0.7)


    # Upsample data
    # xTrain, yTrain = upsample_data(xTrain, yTrain)

    # xTrain, yTrain = upsample_SMOTE(xTrain, yTrain)

    # Drop txId
    if True:
        xTrain = xTrain.drop(['txId'], axis=1)
        xTest = xTest.drop(['txId'], axis=1)


    PCA = False
    if PCA:
        # PCA
        xTrain, xTest = normalize_data(xTrain, xTest)
        pca = apply_PCA(167, xTrain, yTrain)
        xTrain = pca.transform(xTrain)
        xTest = pca.transform(xTest)
        print(xTrain.shape)

    # Just minority
    if False:
        xTest, yTest =  minority_only(xTest, yTest)

        print(xTest.shape)
        print(yTest.shape)
    
    print("Training...")
    SCALE = False
    if SCALE:
        scaler = StandardScaler()
        xTrain = scaler.fit_transform(xTrain)
        xTest = scaler.transform(xTest)

    # Initialize and train model
    model = train(xTrain, yTrain['class'], model_name, grid_search=True)
    print("Done!")

    try:
        print(f"{model_name} best params: {model.best_params_}")
    except:
        pass

    # Only illicit classification
    # xTest = xTest[yTest['class'] == 1]
    # yTest = yTest[yTest['class'] == 1]
    

    print("Predicting...")
    results = predict(model, xTrain, yTrain['class'], xTest, yTest['class'])
    print("Done!")

    for key, value in results.items():
        print(f"{key} : {value}")

    # ROC curve
    metrics.plot_roc_curve(model, xTest, yTest['class'])
    # plt.title("KNN ROC curve")
    # plt.show()

if __name__ == "__main__":
    main()
