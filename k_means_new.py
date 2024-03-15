# open close_pair.csv and see the result
import argparse
from audio_distance import audio_distance
import pandas as pd
from datetime import datetime
import whisper
from util import load_audio_file, mfccs_class_ids_from_files, shuffle_and_split
from util import CLASS_TTS
from util import CLASS_RECORDED
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance


def transpose_mfccs(mfccs):
    max_length = max(mfcc.shape[1] for mfcc in mfccs)
    # Pad MFCCs to have the same shape
    mfccs_padded = [
        np.pad(
            mfcc,
            ((0, 0), (0, max_length - mfcc.shape[1])),
            mode="constant",
            constant_values=0,
        )
        for mfcc in mfccs
    ]

    # Now, mfccs_padded is a list of uniformly shaped arrays, you can stack them
    mfccs_stacked = np.stack(mfccs_padded)

    # transpose
    mfccs_transposed = mfccs_stacked.transpose((0, 2, 1)).reshape(
        mfccs_stacked.shape[0], -1
    )

    return mfccs_transposed


def print_stats(df):
    true_positives = len(
        df[
            (df["test_class"] == CLASS_RECORDED)
            & (df["predicted_class"] == CLASS_RECORDED)
        ]
    )
    print(f"true positives: {true_positives}")

    # calculate true negatives
    true_negatives = len(
        df[(df["test_class"] == CLASS_TTS) & (df["predicted_class"] == CLASS_TTS)]
    )
    print(f"true negatives: {true_negatives}")

    # calculate false positives
    false_positives = len(
        df[(df["test_class"] == CLASS_TTS) & (df["predicted_class"] == CLASS_RECORDED)]
    )
    print(f"false positives: {false_positives}")

    # calculate false negatives
    false_negatives = len(
        df[(df["test_class"] == CLASS_RECORDED) & (df["predicted_class"] == CLASS_TTS)]
    )
    print(f"false negatives: {false_negatives}")

    # calculate accuracy
    accuracy = (true_positives + true_negatives) / len(df)
    print(f"accuracy: {accuracy}")
    # calculate precision
    precision = true_positives / (true_positives + false_positives)
    print(f"precision: {precision}")
    # calculate recall
    recall = true_positives / (true_positives + false_negatives)
    print(f"recall: {recall}")
    # calculate specificity
    specificity = true_negatives / (true_negatives + false_positives)
    print(f"specificity: {specificity}")
    # calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"F1 score: {f1_score}")

    # calculate confusion matrix
    confusion_matrix = pd.crosstab(
        df["test_class"],
        df["predicted_class"],
        rownames=["Actual"],
        colnames=["Predicted"],
    )
    print(confusion_matrix)


def k_means_mfcc(recorded_directory_path, generated_directory_path):
    # get files from directory with eud_distance in the name

    # List all files and directories in the specified path
    recorded_entries = os.listdir(recorded_directory_path)
    generated_entries = os.listdir(generated_directory_path)

    # If you want to list files only
    recorded_files = [
        entry
        for entry in recorded_entries
        if os.path.isfile(os.path.join(recorded_directory_path, entry))
    ]
    generated_files = [
        entry
        for entry in generated_entries
        if os.path.isfile(os.path.join(generated_directory_path, entry))
        and "euc_distance" not in entry
    ]

    # shuffle and split the files
    recorded_files_train, recorded_files_test = shuffle_and_split(recorded_files)
    generated_files_train, generated_files_test = shuffle_and_split(generated_files)

    # open .npy files and get the mfcc
    mfccs_train = []
    class_ids_train = []

    mfccs_train, class_ids_train = mfccs_class_ids_from_files(
        recorded_directory_path,
        recorded_files_train,
        mfccs_train,
        class_ids_train,
        CLASS_RECORDED,
    )

    mfccs_train, class_ids_train = mfccs_class_ids_from_files(
        generated_directory_path,
        generated_files_train,
        mfccs_train,
        class_ids_train,
        CLASS_TTS,
    )

    # open .npy files and get the mfcc
    mfccs_test = []
    class_ids_test = []

    mfccs_test, class_ids_test = mfccs_class_ids_from_files(
        recorded_directory_path,
        recorded_files_test,
        mfccs_test,
        class_ids_test,
        CLASS_RECORDED,
    )

    mfccs_test, class_ids_test = mfccs_class_ids_from_files(
        generated_directory_path,
        generated_files_test,
        mfccs_test,
        class_ids_test,
        CLASS_TTS,
    )

    mfccs_train_transposed = transpose_mfccs(mfccs_train)
    mfccs_test_transposed = transpose_mfccs(mfccs_test)

    # k-means clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(mfccs_train_transposed)
    centers = kmeans.cluster_centers_
    cmap = plt.get_cmap("viridis", max(class_ids_train) + 1)

    # print figure of kmeans clustering
    plt.scatter(
        mfccs_train_transposed[:, 0],
        mfccs_train_transposed[:, 1],
        c=class_ids_train,
        cmap=cmap,
    )
    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)
    # plt.colorbar(ticks=range(max(class_ids_train) + 1))
    plt.colorbar(ticks=[0, 1])
    plt.savefig("kmeans_binary.png")
    plt.tight_layout()
    plt.close()
    return (
        centers,
        mfccs_train_transposed,
        class_ids_train,
        mfccs_test_transposed,
        class_ids_test,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute distance between two audio files."
    )
    parser.add_argument(
        "recorded_directory_path",
        type=str,
        help="The directory path to the recorded audio files",
    )
    parser.add_argument(
        "generated_directory_path",
        type=str,
        help="The directory path to the generated audio files",
    )
    args = parser.parse_args()
    centers, mfccs_train, class_ids_train, mfccs_test, class_ids_test = k_means_mfcc(
        args.recorded_directory_path, args.generated_directory_path
    )
    results = []
    # save results to a file with two columns, one for the test class and the other for the predicted class

    # for each test class, compute the distance to the two centers
    for i in range(len(mfccs_test[:, 0])):
        distance_0 = distance.euclidean(
            [mfccs_test[:, 0][i], mfccs_test[:, 1][i]],
            [centers[:, 0][0], centers[:, 1][0]],
        )
        distance_1 = distance.euclidean(
            [mfccs_test[:, 0][i], mfccs_test[:, 1][i]],
            [centers[:, 0][1], centers[:, 1][1]],
        )
        if distance_0 < distance_1:
            results.append([class_ids_test[i], CLASS_RECORDED])
        else:
            results.append([class_ids_test[i], CLASS_TTS])
    # save results to results_accuracy.csv
    df = pd.DataFrame(results, columns=["test_class", "predicted_class"])
    df.to_csv("results_accuracy.csv", index=False)

    print_stats(df)


# centers = k_means_mfcc(
#     "k_generated_mfcc/2024-03-06--21-44-04", "generated_mfcc/2024-03-03--13-54-02"
# )

# print(centers)

## main function
if __name__ == "__main__":
    print("__main__")
    main()
    # centers, mfccs_train, class_ids_train, mfccs_test, class_ids_test = k_means_mfcc(
    #     "k_generated_mfcc/2024-03-06--21-44-04", "generated_mfcc/2024-03-03--13-54-02"
    # )
    # # print dimensions of centers
    # print(centers.shape)
    # print(len(mfccs_train))

    # center_recorded = centers[:, 0]
    # center_tts = centers[:, 1]

    # results = []
    # # save results to a file with two columns, one for the test class and the other for the predicted class

    # # for each test class, compute the distance to the two centers
    # for i in range(len(mfccs_test[:, 0])):
    #     distance_0 = distance.euclidean(
    #         [mfccs_test[:, 0][i], mfccs_test[:, 1][i]],
    #         [centers[:, 0][0], centers[:, 1][0]],
    #     )
    #     distance_1 = distance.euclidean(
    #         [mfccs_test[:, 0][i], mfccs_test[:, 1][i]],
    #         [centers[:, 0][1], centers[:, 1][1]],
    #     )
    #     if distance_0 < distance_1:
    #         results.append([class_ids_test[i], CLASS_RECORDED])
    #     else:
    #         results.append([class_ids_test[i], CLASS_TTS])
    # # save results to results_accuracy.csv
    # df = pd.DataFrame(results, columns=["test_class", "predicted_class"])
    # df.to_csv("results_accuracy.csv", index=False)

    # print_stats(df)
