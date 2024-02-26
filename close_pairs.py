import itertools
from Levenshtein import distance
import pandas as pd

PHRASE_COLUMN = "sentence"
DELIMITER = "\t"

INPUT_FILE_PATH = (
    "cv-corpus-16.1-2023-12-06-en/cv-corpus-16.1-2023-12-06/en/validated.tsv"
)


OUTPUT_FILE_PATH = (
    "cv-corpus-16.1-2023-12-06-en/cv-corpus-16.1-2023-12-06/en/close_pairs.csv"
)


def find_close_phrase_pairs(phrases, max_distance):
    """
    Finds and returns pairs of phrases with Levenshtein distances below max_distance.

    :param phrases: A list of phrases to compare.
    :param max_distance: The maximum Levenshtein distance to consider as 'close'.
    :return: A list of tuples, where each tuple contains a pair of phrases and their distance.
    """
    close_pairs = []
    # Compare each unique pair of phrases
    i = 0
    for phrase1, phrase2 in itertools.combinations(phrases, 2):
        dist = distance(phrase1, phrase2)
        if dist <= max_distance and dist > 0:
            close_pairs.append((i, phrase1, phrase2, dist))
            i += 1
    return close_pairs


# Load the dataset
df = pd.read_csv(
    INPUT_FILE_PATH,
    delimiter=DELIMITER,
)
print(df.head())
print(df.columns)
# Example usage
phrases = df[PHRASE_COLUMN].tolist()
max_distance = 3  # Define 'short distance'
close_pairs = find_close_phrase_pairs(phrases, max_distance)

# save the close_pairs to csv
import csv

with open(OUTPUT_FILE_PATH, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "phrase1", "phrase2", "distance"])
    for pair in close_pairs:
        writer.writerow(pair)

for pair in close_pairs:
    print(pair)
