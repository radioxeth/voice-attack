import itertools
from Levenshtein import distance


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


### load cv-valid-dev.csv
# Load the dataset
import pandas as pd

# Load the dataset
df = pd.read_csv("cv-valid-dev/cv-valid-dev.csv")
print(df["text"].head())

# Example usage
phrases = df["text"].tolist()
max_distance = 3  # Define 'short distance'
close_pairs = find_close_phrase_pairs(phrases, max_distance)

# save the close_pairs to csv
import csv

with open("cv-valid-dev/close_pairs.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "phrase1", "phrase2", "distance"])
    for pair in close_pairs:
        writer.writerow(pair)

for pair in close_pairs:
    print(pair)
