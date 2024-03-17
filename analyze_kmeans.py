# analyze_kmeans.py

# generate a ROC curve using the results
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd

FILENAME = "kmeans_final/results_accuracy.csv"

df = pd.read_csv(FILENAME)
print(df.head())


# for each n_mfcc find the mean of each accuracy metric
# and the standard deviation of each accuracy metric
n_mfccs = df["n_mfcc"].unique()
means = []
for n_mfcc in n_mfccs:
    df_n_mfcc = df[df["n_mfcc"] == n_mfcc]
    print(f"n_mfcc: {n_mfcc}")

    # append mean and std to means for true_positives, true_negatives, false_positives, false_negatives, accuracy, precision, recall, specificity, f1_score
    means.append(
        [
            n_mfcc,
            # df_n_mfcc["true_positives"].mean(),
            # df_n_mfcc["true_positives"].std(),
            # df_n_mfcc["true_negatives"].mean(),
            # df_n_mfcc["true_negatives"].std(),
            # df_n_mfcc["false_positives"].mean(),
            # df_n_mfcc["false_positives"].std(),
            # df_n_mfcc["false_negatives"].mean(),
            # df_n_mfcc["false_negatives"].std(),
            df_n_mfcc["accuracy"].mean().round(4),
            df_n_mfcc["accuracy"].std().round(4),
            df_n_mfcc["precision"].mean().round(4),
            df_n_mfcc["precision"].std().round(4),
            df_n_mfcc["recall"].mean().round(4),
            df_n_mfcc["recall"].std().round(4),
            df_n_mfcc["specificity"].mean().round(4),
            df_n_mfcc["specificity"].std().round(4),
            df_n_mfcc["f1_score"].mean().round(4),
            df_n_mfcc["f1_score"].std().round(4),
        ]
    )

# plot the mean accuracy metrics for each n_mfcc
df_means = pd.DataFrame(
    means,
    columns=[
        "n_mfcc",
        # "true_positives_mean",
        # "true_positives_std",
        # "true_negatives_mean",
        # "true_negatives_std",
        # "false_positives_mean",
        # "false_positives_std",
        # "false_negatives_mean",
        # "false_negatives_std",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "specificity_mean",
        "specificity_std",
        "f1_score_mean",
        "f1_score_std",
    ],
)
df_means.to_csv("kmeans_final/results_accuracy_means.csv", index=False)


# plot the mean accuracy metrics for each n_mfcc
fig, ax = plt.subplots()
ax.errorbar(
    df_means["n_mfcc"],
    df_means["accuracy_mean"],
    yerr=df_means["accuracy_std"],
    fmt="o",
)
ax.set_xlabel("n_mfcc")
ax.set_ylabel("accuracy")
plt.savefig("kmeans_final/accuracy_vs_n_mfcc.png")
plt.close()


# plot the roc curve
fig, ax = plt.subplots()
lw = 2  # Line width for the plots
for n_mfcc in n_mfccs:
    KMEANS_FILENAME = f"kmeans_final/results_mfcc{n_mfcc}_round0.csv"

    # Read the results from the csv file
    df_res = pd.read_csv(KMEANS_FILENAME)

    # Get the true class and the predicted class
    y_true = df_res["test_class"]
    y_pred = df_res["predicted_class"]

    # Generate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Add the ROC curve to the plot
    ax.plot(fpr, tpr, lw=lw, label=f"ROC curve MFCC={n_mfcc} (area = {roc_auc:.2f})")

# Configure the plot with common settings
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic")
# ax.legend(loc="lower right", fontsize="small")

# Save the figure with all ROC curves
plt.savefig("kmeans_final/roc_curve_all_mfcc.png")
plt.close()


print(np.max(df["accuracy"]))
print(df[df["accuracy"] == np.max(df["accuracy"])])

print(np.max(df_means["accuracy_mean"]))
print(df_means[df_means["accuracy_mean"] == np.max(df_means["accuracy_mean"])])
