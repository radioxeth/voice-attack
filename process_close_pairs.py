# open close_pair.csv and see the result
from audio_distance import audio_distance
import pandas as pd
from datetime import datetime

DIRECTORY = "cv-valid-dev/"


# Load the close_pairs dataset
df_close_pairs = pd.read_csv(f"{DIRECTORY}close_pairs.csv")
print(df_close_pairs.head())

# load cv-valid-dev.csv
df = pd.read_csv(f"{DIRECTORY}cv-valid-dev.csv")
print(df.head())

now = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

# loop through the close_pairs dataset and print the corresponding text from the cv-valid-dev.csv dataset
for index, row in df_close_pairs.iterrows():
    filename1 = df[df["text"] == row["phrase1"]]["filename"].values[0]
    text1 = df[df["text"] == row["phrase1"]]["text"].values[0]
    filename2 = df[df["text"] == row["phrase2"]]["filename"].values[0]
    text2 = df[df["text"] == row["phrase2"]]["text"].values[0]
    id = row["id"]
    print(f"{filename1}: {text1}")
    print(f"{filename2}: {text2}")
    print("\n")
    audio_distance(f"{DIRECTORY}{filename1}", f"{DIRECTORY}{filename2}", id=id, now=now)
