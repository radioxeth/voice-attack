# process naive bayes
import pandas as pd
import os
from k_means_new import k_means
from util import CLASS_RECORDED, CLASS_TTS
from k_means_new import get_accuracy_stats

#### Generated Audio Params
DIRECTORY_GEN = "mfcc_generated"

#### Recorded Audio Params
DIRECTORY_REC = "mfcc_recorded"

#### output directory
OUTPUT_DIR = "kmeans"

#### Results filename
FILENAME = f"{OUTPUT_DIR}/results_accuracy.csv"

results = []
# # loop through the recorded and generated mfcc directories
for n_mfcc in range(1, 30):
    # write to csv file results.csv
    generated_dir = f"{DIRECTORY_GEN}/final_{n_mfcc}"
    recorded_dir = f"{DIRECTORY_REC}/final_{n_mfcc}"
    for round in range(1, 10):
        print(f"n_mfcc: {n_mfcc}, round: {round}")
        centers, true_positives, true_negatives, false_positives, false_negatives = (
            k_means(
                generated_dir,
                recorded_dir,
                n_mfcc=n_mfcc,
                round=round,
                output_dir=OUTPUT_DIR,
            )
        )
        # get accuracy stats
        accuracy, precision, recall, specificity, f1_score = get_accuracy_stats(
            true_positives, true_negatives, false_positives, false_negatives
        )
        results.append(
            [
                n_mfcc,
                round,
                centers[0][0],
                centers[0][1],
                centers[1][0],
                centers[1][1],
                true_positives,
                true_negatives,
                false_positives,
                false_negatives,
                accuracy,
                precision,
                recall,
                specificity,
                f1_score,
            ]
        )
