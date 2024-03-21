# analyze_naive_bayes.py

# generate a ROC curve using the results
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd

OUTPUT_DIR = "naive_bayes_final_vary_pca"

FILENAME = f"{OUTPUT_DIR}/results_accuracy.csv"

df = pd.read_csv(FILENAME)
print(df.head())

#### output directory

# for each n_mfcc find the mean of each accuracy metric
# and the standard deviation of each accuracy metric
n_mfccs = df["n_mfcc"].unique()
means = []
for n_mfcc in n_mfccs:
    df_n_mfcc = df[df["n_mfcc"] == n_mfcc]
    print(f"n_mfcc: {n_mfcc}")
    print(df_n_mfcc["accuracy"].mean(), df_n_mfcc["accuracy"].std())
    # append mean and std to means for true_positives, true_negatives, false_positives, false_negatives, accuracy, precision, recall, specificity, f1_score
    means.append(
        [
            n_mfcc,
            df_n_mfcc["true_positives"].sum(),
            df_n_mfcc["true_negatives"].sum(),
            df_n_mfcc["false_positives"].sum(),
            df_n_mfcc["false_negatives"].sum(),
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

# scatter plot of mfcc vs accuracy
# color = iter(plt.cm.rainbow(np.linspace(0, 1, len(n_mfccs))))
plt.figure()
for n_mfcc in n_mfccs:
    # print(f"n_mfcc: {n_mfcc}")
    df_n_mfcc = df[df["n_mfcc"] == n_mfcc]
    plt.scatter(df_n_mfcc["n_mfcc"], df_n_mfcc["accuracy"], label=f"{n_mfcc} MFCCs")


# start the color scheme at the second color

plt.xlim(0, 41)
plt.ylim(0, 1)
plt.xlabel("Number of MFCCs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of MFCCs\nNaive Bayes")

# add legend with small font
# plt.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_vs_mfcc.png")
plt.close()


# plot the mean accuracy metrics for each n_mfcc
df_means = pd.DataFrame(
    means,
    columns=[
        "n_mfcc",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
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
df_means.to_csv(f"{OUTPUT_DIR}/results_accuracy_means.csv", index=False)

print(df_means.head())
for index, row in df_means.iterrows():
    print(row)
    tp = row[1]
    tn = row[2]
    fp = row[3]
    fn = row[4]
    print(tp, tn, fp, fn)
    # fp = mean["false_positives"]
    # fn = mean["false_negatives"]
    # print a labeled table of the confusion matrix
    print("##################################")
    print("Confusion Matrix")
    print("Predicted", "Recorded", "Generated")
    print("Actual")
    print("Recorded", tp, fn)
    print("Generated", fp, tn)
    print("##################################")


# plot the mean accuracy metrics for each n_mfcc
fig, ax = plt.subplots()
ax.errorbar(
    df_means["n_mfcc"],
    df_means["accuracy_mean"],
    yerr=df_means["accuracy_std"],
    fmt="o",
    label="Accuracy",
)
ax.set_xlabel("Number of MFCCs")
ax.set_ylabel("Accuracy")
ax.set_title("Mean Accuracy vs Number of MFCCs\nNaive Bayes")
ax.set_ylim(0, 1)
ax.set_xlim(0, 41)
plt.tight_layout()
plt.legend(loc="lower right")
plt.savefig(f"{OUTPUT_DIR}/mean_accuracy_vs_n_mfcc.png")
plt.close()


# plot the mean accuracy metrics for each n_mfcc ALL

fig, ax = plt.subplots()
ax.errorbar(
    df_means["n_mfcc"],
    df_means["accuracy_mean"],
    yerr=df_means["accuracy_std"],
    fmt="o",
    label="Accuracy",
)
ax.errorbar(
    df_means["n_mfcc"],
    df_means["precision_mean"],
    yerr=df_means["precision_std"],
    fmt="x",
    label="Precision",
)
ax.errorbar(
    df_means["n_mfcc"],
    df_means["recall_mean"],
    yerr=df_means["recall_std"],
    fmt="v",
    label="Recall",
)
ax.errorbar(
    df_means["n_mfcc"],
    df_means["specificity_mean"],
    yerr=df_means["specificity_std"],
    fmt="d",
    label="Specificity",
)
ax.set_xlabel("Number of MFCCs")
ax.set_ylabel("Performance Metric Value")
ax.set_title("Performance Metrics vs Number of MFCCs\nNaive Bayes")
ax.set_ylim(0, 1)
ax.set_xlim(0, 41)
plt.tight_layout()
plt.legend(loc="lower right")
plt.savefig(f"{OUTPUT_DIR}/mean_accuracy_vs_n_mfcc_all.png")
plt.close()


# plot the roc curve
fig, ax = plt.subplots()
lw = 2  # Line width for the plots
for n_mfcc in n_mfccs:
    NB_FILENAME = f"{OUTPUT_DIR}/results_mfcc{n_mfcc}_round0.csv"

    # Read the results from the csv file
    df_res = pd.read_csv(NB_FILENAME)

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
ax.set_title("Receiver Operating Characteristic\nNaive Bayes")
# ax.legend(loc="lower right", fontsize="small")

# Save the figure with all ROC curves
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve_all_mfcc.png")
plt.close()


max_accuracy = df["accuracy"].max()
print("accuracy", max_accuracy)
print(df[df["accuracy"] == max_accuracy])

max_mean_accuracy = df_means["accuracy_mean"].max()
print("mean accuracy", max_mean_accuracy)
print(df_means[df_means["accuracy_mean"] == max_mean_accuracy])

max_mean_accuracy_std = df_means[df_means["accuracy_mean"] == max_mean_accuracy][
    "accuracy_std"
].values[0]
print("mean accuracy std", max_mean_accuracy_std)
