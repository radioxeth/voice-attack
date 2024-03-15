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
    pair_ids = []
    for file in files:
        mfcc = np.load(f"{directory_path}/{file}")
        mfccs.append(mfcc)
        pair_id = file.split("-")[0]
        pair_ids.append(int(pair_id))

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
    kmeans = KMeans(n_clusters=2, random_state=0).fit(mfccs_transposed)
    print(mfccs_transposed)
    print(kmeans)

    cmap = plt.get_cmap("viridis", max(pair_ids) + 1)
    print(cmap)
    # print figure of kmeans clustering
    plt.scatter(mfccs_transposed[:, 0], mfccs_transposed[:, 1], c=pair_ids, cmap=cmap)
    # add legend with pair_ids and color
    for i, pair_id in enumerate(pair_ids):
        plt.text(
            mfccs_transposed[i, 0],
            mfccs_transposed[i, 1],
            str(pair_id),
            color=cmap(pair_id),
        )
    # build a list of the pair_ids and their color

    centers = kmeans.cluster_centers_

    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)
    # plt.colorbar(ticks=range(max(pair_ids) + 1))
    plt.savefig("kmeans.png")


k_means_mfcc("k_generated_mfcc/2024-03-06--21-44-04")
