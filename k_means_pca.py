# open close_pair.csv and see the result
import argparse

from sklearn.impute import SimpleImputer
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
from sklearn.decomposition import PCA


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


def get_accuracy_stats(
    true_positives, true_negatives, false_positives, false_negatives
):
    # calculate accuracy
    accuracy = (true_positives + true_negatives) / (
        true_positives + false_positives + false_negatives + true_negatives
    )

    # calculate precision
    precision = true_positives / (true_positives + false_positives)

    # calculate recall
    recall = true_positives / (true_positives + false_negatives)

    # calculate specificity
    specificity = true_negatives / (true_negatives + false_positives)

    # calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, specificity, f1_score


def get_confusion_stats(df):
    true_positives = len(
        df[
            (df["test_class"] == CLASS_RECORDED)
            & (df["predicted_class"] == CLASS_RECORDED)
        ]
    )

    # calculate true negatives
    true_negatives = len(
        df[(df["test_class"] == CLASS_TTS) & (df["predicted_class"] == CLASS_TTS)]
    )

    # calculate false positives
    false_positives = len(
        df[(df["test_class"] == CLASS_TTS) & (df["predicted_class"] == CLASS_RECORDED)]
    )

    # calculate false negatives
    false_negatives = len(
        df[(df["test_class"] == CLASS_RECORDED) & (df["predicted_class"] == CLASS_TTS)]
    )

    return true_positives, true_negatives, false_positives, false_negatives


def pad_mfccs(mfccs):
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

    return mfccs_padded


def pca_mfccs(mfccs, n_components=2):
    mfccs = np.array(mfccs)
    mfccs = mfccs.reshape(mfccs.shape[0], -1)

    # Initialize and apply imputation
    imputer = SimpleImputer(strategy="mean")
    mfccs_imputed = imputer.fit_transform(mfccs)

    # Standardization
    # scaler = StandardScaler(with_mean=False)
    # X_train = scaler.fit_transform(mfccs_imputed)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(mfccs_imputed)

    return X_pca


def k_means_mfcc(recorded_directory_path, generated_directory_path, n_mfcc):
    mfccs_train = []
    class_ids_train = []
    mfccs_test = []
    class_ids_test = []

    # List all files and directories in the specified path
    recorded_entries = os.listdir(recorded_directory_path)
    generated_entries = os.listdir(generated_directory_path)

    # If you want to list files only
    recorded_files = [
        entry
        for entry in recorded_entries
        if os.path.isfile(os.path.join(recorded_directory_path, entry))
        and ".npy" in entry
    ]

    generated_files = [
        entry
        for entry in generated_entries
        if os.path.isfile(os.path.join(generated_directory_path, entry))
        and ".npy" in entry
    ]

    # shuffle and split the files
    recorded_files_train, recorded_files_test = shuffle_and_split(recorded_files)
    generated_files_train, generated_files_test = shuffle_and_split(generated_files)

    # open .npy files and get the mfcc
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

    mfccs_train = pad_mfccs(mfccs_train)
    mfccs_train = pca_mfccs(mfccs_train, n_mfcc)

    mfccs_test = pad_mfccs(mfccs_test)
    mfccs_test = pca_mfccs(mfccs_test, n_mfcc)

    print(mfccs_train.shape, mfccs_test.shape)
    # k-means clustering
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(mfccs_train)
    centers = kmeans.cluster_centers_

    return (
        centers,
        mfccs_train,
        class_ids_train,
        mfccs_test,
        class_ids_test,
    )


def print_kmeans_figure(centers, mfccs_train, class_ids_train, n_mfcc, figure_name):
    cmap = plt.get_cmap("viridis", max(class_ids_train) + 1)
    plt.scatter(
        mfccs_train[:, 0],
        mfccs_train[:, 1],
        c=class_ids_train,
        cmap=cmap,
    )
    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)
    plt.colorbar(ticks=[0, 1])
    # plt.legend()
    plt.title(f"K-means clustering\n{n_mfcc} MFCCs\n")
    plt.tight_layout()
    plt.savefig(figure_name)
    plt.close()


def k_means(
    recorded_directory_path, generated_directory_path, n_mfcc, round, output_dir
):
    # train k-means model
    centers, mfccs_train, class_ids_train, mfccs_test, class_ids_test = k_means_mfcc(
        recorded_directory_path, generated_directory_path, n_mfcc
    )
    os.makedirs(f"{output_dir}", exist_ok=True)

    # print figure of kmeans clustering
    figure_name = f"{output_dir}/kmeans_mfcc{n_mfcc}_round{round}.png"
    print_kmeans_figure(centers, mfccs_train, class_ids_train, n_mfcc, figure_name)

    # save results to a file with two columns, one for the test class and the other for the predicted class
    results = []
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
    df.to_csv(f"{output_dir}/results_mfcc{n_mfcc}_round{round}.csv", index=False)
    true_positives, true_negatives, false_positives, false_negatives = (
        get_confusion_stats(df)
    )
    accuracy, precision, recall, specificity, f1_score = get_accuracy_stats(
        true_positives, true_negatives, false_positives, false_negatives
    )
    # print confusion matrix
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Specificity: {specificity}")
    print(f"F1 Score: {f1_score}")
    print(f"True Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")

    # calculate confusion matrix
    print(
        f"Confusion Matrix: \n{df.groupby(['test_class', 'predicted_class']).size().unstack(fill_value=0)}"
    )

    # if accuracy is greater than .5 save
    if accuracy > 0.5:
        df = pd.DataFrame(results, columns=["test_class", "predicted_class"])
        df.to_csv(f"{output_dir}/results_mfcc{n_mfcc}_round{round}.csv", index=False)
        return (
            [
                [centers[:, 0][0], centers[:, 1][0]],
                [centers[:, 0][1], centers[:, 1][1]],
            ],
            true_positives,
            true_negatives,
            false_positives,
            false_negatives,
        )

    # else flip the centers and save
    results = []
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
        if distance_0 > distance_1:
            results.append([class_ids_test[i], CLASS_RECORDED])
        else:
            results.append([class_ids_test[i], CLASS_TTS])

    df = pd.DataFrame(results, columns=["test_class", "predicted_class"])
    df.to_csv(f"{output_dir}/results_mfcc{n_mfcc}_round{round}.csv", index=False)
    true_positives, true_negatives, false_positives, false_negatives = (
        get_confusion_stats(df)
    )
    df = pd.DataFrame(results, columns=["test_class", "predicted_class"])
    df.to_csv(f"{output_dir}/results_mfcc{n_mfcc}_round{round}.csv", index=False)
    return (
        [
            [centers[:, 0][1], centers[:, 1][1]],
            [centers[:, 0][0], centers[:, 1][0]],
        ],
        true_positives,
        true_negatives,
        false_positives,
        false_negatives,
    )


## main function
if __name__ == "__main__":
    print("__main__")
    parser = argparse.ArgumentParser(
        description="K means for mfccs and pca of recorded and generated audio files"
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
    parser.add_argument(
        "n_mfcc", type=int, help="Number of mfcc features to extract", default=13
    )
    parser.add_argument("round", type=int, help="round of the model run", default=0)
    parser.add_argument(
        "output_dir", type=str, help="output directory", default="kmeans"
    )
    args = parser.parse_args()
    k_means(
        args.recorded_directory_path,
        args.generated_directory_path,
        args.n_mfcc,
        args.round,
        args.output_dir,
    )
