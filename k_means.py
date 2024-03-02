# open close_pair.csv and see the result
from audio_distance import audio_distance
import pandas as pd
from datetime import datetime
import whisper
from util import load_audio_file
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def k_means_mfcc(directory_path):
    # get files from directory with eud_distance in the name

    # List all files and directories in the specified path
    entries = os.listdir(directory_path)

    # If you want to list files only
    files = [
        entry
        for entry in entries
        if os.path.isfile(os.path.join(directory_path, entry))
        and "euc_distance" in entry
    ]
    # print(files)
    # open .npy files and get the mfcc
    mfccs = []
    for file in files:
        mfcc = np.load(f"{directory_path}/{file}")
        mfccs.append(mfcc)

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

    # k-means clustering
    kmeans = KMeans(n_clusters=40, random_state=0).fit(mfccs_transposed)
    print(kmeans)
    # print figure of kmeans clustering
    plt.scatter(
        mfccs_transposed[:, 0], mfccs_transposed[:, 1], c=kmeans.labels_, cmap="viridis"
    )
    centers = kmeans.cluster_centers_
    # add legend
    # plt.legend()
    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)
    plt.savefig("kmeans.png")


k_means_mfcc("generated_mfcc/2024-03-01--19-43-03")
