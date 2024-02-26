### open file as csv

from Levenshtein import distance
import pandas as pd

INPUT_FILE_PATH = "generated_audio/tts2024-02-23--21-16-56/results.csv"
DELIMITER = ","

# Load the dataset
df = pd.read_csv(
    INPUT_FILE_PATH,
    delimiter=DELIMITER,
)

print(df.head())

# loop through the dataset and print the corresponding text from the cv-valid-dev.csv dataset

for index, row in df.iterrows():
    for symbol in "_-!'(),.:;?":
        text = row["text"].replace(symbol, "").lower().strip()
        transcript = row["transcript"].replace(symbol, "").lower().strip()

    tts_distance = distance(text, transcript)
    # append distance to row
    df.loc[index, "tts_distance"] = tts_distance
    df.loc[index, "text"] = text
    df.loc[index, "transcript"] = transcript

    print(f"{tts_distance} -- {text} :: {transcript}")

### Write to file
df.to_csv(INPUT_FILE_PATH, index=False)
