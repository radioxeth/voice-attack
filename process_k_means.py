import pandas as pd
from util import CLASS_RECORDED, CLASS_TTS
from util import get_accuracy_stats
from k_means import k_means

import matplotlib.pyplot as plt

# loop from id 1 to 40 to iterate through the mfcc_generated/final_{id} and mfcc_recorded/final_{id} directories

#### Generated Audio Params
DIRECTORY_GEN = "mfcc_generated"

#### Recorded Audio Params
DIRECTORY_REC = "mfcc_recorded"

#### output directory
OUTPUT_DIR = "kmeans_final"

#### Results filename
FILENAME = f"{OUTPUT_DIR}/results_accuracy.csv"

results = []
# # loop through the recorded and generated mfcc directories
for n_mfcc in range(1, 41):
    # write to csv file results.csv
    generated_dir = f"{DIRECTORY_GEN}/final_{n_mfcc}"
    recorded_dir = f"{DIRECTORY_REC}/final_{n_mfcc}"
    for round in range(0, 10):
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

# save results to results_accuracy.csv
df = pd.DataFrame(
    results,
    columns=[
        "n_mfcc",
        "round",
        "center_0_x",
        "center_0_y",
        "center_1_x",
        "center_1_y",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
    ],
)
df.to_csv(FILENAME, index=False)
