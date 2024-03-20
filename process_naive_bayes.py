# process naive bayes
import pandas as pd
import os
from naive_bayes import naive_bayes
from util import CLASS_RECORDED, CLASS_TTS
from k_means import get_accuracy_stats

#### Generated Audio Params
DIRECTORY_GEN = "mfcc_generated"

#### Recorded Audio Params
DIRECTORY_REC = "mfcc_recorded"

#### output directory
OUTPUT_DIR = "naive_bayes_final_vary_pca_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#### Results filename
FILENAME = f"{OUTPUT_DIR}/results_accuracy.csv"


results = []
# # loop through the recorded and generated mfcc directories
for n_mfcc in range(1, 2):
    # write to csv file results.csv
    generated_dir = f"{DIRECTORY_GEN}/final_{n_mfcc}"
    recorded_dir = f"{DIRECTORY_REC}/final_{n_mfcc}"
    for round in range(0, 10):
        print(f"n_mfcc: {n_mfcc}, round: {round}")
        true_positives, true_negatives, false_positives, false_negatives = naive_bayes(
            generated_dir,
            recorded_dir,
            n_mfcc=n_mfcc,
            round=round,
            output_dir=OUTPUT_DIR,
        )
        # get accuracy stats
        accuracy, precision, recall, specificity, f1_score = get_accuracy_stats(
            true_positives, true_negatives, false_positives, false_negatives
        )

        results.append(
            [
                n_mfcc,
                round,
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
