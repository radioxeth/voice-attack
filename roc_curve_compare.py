# roc_curve_compare.py
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import auc, roc_curve

OUTPUT_DIR = "compare_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_MEANS_DIR = "kmeans_final"
mean_accuracy_filename_k_means = f"{K_MEANS_DIR}/results_accuracy_means.csv"
accuracy_filename_k_means = f"{K_MEANS_DIR}/results_accuracy.csv"

NAIVE_BAYES_DIR = "naive_bayes_final_vary_pca"
mean_accuracy_filename_naive_bayes = f"{NAIVE_BAYES_DIR}/results_accuracy_means.csv"
accuracy_filename_naive_bayes = f"{NAIVE_BAYES_DIR}/results_accuracy.csv"


# read in the mean accuracy files
df_k_means = pd.read_csv(mean_accuracy_filename_k_means)
max_mean_accuracy_k_means = df_k_means["accuracy_mean"].max()

df_naive_bayes = pd.read_csv(mean_accuracy_filename_naive_bayes)
max_mean_accuracy_naive_bayes = df_naive_bayes["accuracy_mean"].max()


# plot the accuracy of k-means and naive bayes
# labeling the k-means and naive bayes data
fig, ax = plt.subplots()
ax.errorbar(
    df_k_means["n_mfcc"],
    df_k_means["accuracy_mean"],
    yerr=df_k_means["accuracy_std"],
    fmt="o",
    label="K-means",
)
ax.errorbar(
    df_naive_bayes["n_mfcc"],
    df_naive_bayes["accuracy_mean"],
    yerr=df_naive_bayes["accuracy_std"],
    fmt="x",
    label="Naive Bayes",
)


# plt.scatter(df_k_means["n_mfcc"], df_k_means["accuracy_mean"], label="K-means")
# plt.scatter(
#     df_naive_bayes["n_mfcc"], df_naive_bayes["accuracy_mean"], label="Naive Bayes"
# )
plt.xlabel("Number of MFCCs")
plt.ylabel("Accuracy")
plt.title("Mean Accuracy vs Number of MFCCs\nK-means vs Naive Bayes")
plt.legend()
plt.ylim(0, 1)
plt.xlim(0, 41)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_vs_mfcc_compare.png")
plt.close()

# plot the roc curve for the best k-means and naive bayes models
# read in the accuracy files
df_k_means_all = pd.read_csv(accuracy_filename_k_means)
df_naive_bayes_all = pd.read_csv(accuracy_filename_naive_bayes)

max_accuracy_k_means = df_k_means_all["accuracy"].max()
max_accuracy_naive_bayes = df_naive_bayes_all["accuracy"].max()

# get the best k-means and naive bayes models
best_k_means = df_k_means_all[df_k_means_all["accuracy"] == max_accuracy_k_means]

best_naive_bayes = df_naive_bayes_all[
    df_naive_bayes_all["accuracy"] == max_accuracy_naive_bayes
]

print(f"best_k_means:\n {best_k_means}")
print(f"best_naive_bayes:\n {best_naive_bayes}")

# calculate the TPR and FPR for the best k-means and naive bayes models
tpr_k_means = best_k_means["true_positives"] / (
    best_k_means["true_positives"] + best_k_means["false_negatives"]
)
fpr_k_means = best_k_means["false_positives"] / (
    best_k_means["false_positives"] + best_k_means["true_negatives"]
)

tpr_naive_bayes = best_naive_bayes["true_positives"] / (
    best_naive_bayes["true_positives"] + best_naive_bayes["false_negatives"]
)
fpr_naive_bayes = best_naive_bayes["false_positives"] / (
    best_naive_bayes["false_positives"] + best_naive_bayes["true_negatives"]
)

# plot the roc curve
fig, ax = plt.subplots()
lw = 2  # Line width for the plots

k_means_mfcc = best_k_means["n_mfcc"].values[0]
naive_bayes_mfcc = best_naive_bayes["n_mfcc"].values[0]

k_means_round = best_k_means["round"].values[0]
naive_bayes_round = best_naive_bayes["round"].values[0]

filenames = [
    f"{K_MEANS_DIR}/results_mfcc{k_means_mfcc}_round{k_means_round}.csv",
    f"{NAIVE_BAYES_DIR}/results_mfcc{naive_bayes_mfcc}_round{naive_bayes_round}.csv",
]

for filename in filenames:
    if K_MEANS_DIR in filename:
        method = "K-means"
        n_mfcc = k_means_mfcc
        line = "solid"
    else:
        method = "Naive Bayes"
        n_mfcc = naive_bayes_mfcc
        line = "dashed"
    # Read the results from the csv file
    df_res = pd.read_csv(filename)

    # Get the true class and the predicted class
    y_true = df_res["test_class"]
    y_pred = df_res["predicted_class"]

    # Generate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Add the ROC curve to the plot
    ax.plot(
        fpr,
        tpr,
        lw=lw,
        linestyle=f"{line}",
        label=f"{method} MFCC={n_mfcc} (area = {roc_auc:.2f})",
    )

# Configure the plot with common settings
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic\nK-means vs Naive Bayes")
ax.legend(loc="lower right", fontsize="small")
plt.tight_layout()
# Save the figure with all ROC curves
plt.savefig(f"{OUTPUT_DIR}/roc_curve_compare.png")
plt.close()
