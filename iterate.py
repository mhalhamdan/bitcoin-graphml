from pandas.core.frame import DataFrame
from main import run, run_kfold

import matplotlib.pyplot as plt

import numpy as np

def main():

    # models = ["dt", "knn"]
    # models = ["knn"]
    models = ["knn"]

    options = [{"pca": True, "correlation": False}, {"pca": False, "correlation": True}, {"pca": False, "correlation": False}]

    # options = [{"pca": False, "correlation": False}]

    upsample = ["+U", "SMOTE"]

    upsample = ["+U", ""]

    # data = {"dt": {"recall": [], "accuracy": [], "labels": []}, "knn": {"recall": [], "accuracy": [], "labels": []}}

    data = {x: {"f1_score": [], "accuracy": [], "labels": []} for x in models}

    for m in models:

        for o in options:

            for u in upsample:

                title = "Full"

                if o["pca"]:
                    title = "PCA"
                elif o["correlation"]:
                    title = "Corr"
                if u:
                    title = f"{title} {u}"

                results = run_kfold(m, o["correlation"], o["pca"], u, title=title)

                data[m]["f1_score"].append(results["f1_score"])
                data[m]["accuracy"].append(results["test_acc"])
                data[m]["labels"].append(title)
    
        labels = data[m]["labels"]
        recall = data[m]["f1_score"]
        accuracy = data[m]["accuracy"]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, recall, width, label='F1 score')
        rects2 = ax.bar(x + width/2, accuracy, width, label='Accuracy')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        title = f"{m} F1 score and accuracy (K-fold = 3)"

        ax.set_title(title)
        
        if False:
            labels.insert(1, "")
            labels.insert(1, "")
            labels.insert(1, "")
            labels.insert(0, "")
            labels.insert(0, "")

        if True:
            labels.insert(0, "test")
        
        ax.set_xticklabels(labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        
        fig.savefig(title)




if __name__ == "__main__":
    main()