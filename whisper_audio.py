# open close_pair.csv and see the result
from audio_distance import audio_distance
import pandas as pd
from datetime import datetime
import whisper
from util import load_audio_file
import torch

DIRECTORY = "cv-valid-dev/"

# Load the close_pairs dataset
df_close_pairs = pd.read_csv(f"{DIRECTORY}close_pairs.csv")
print(df_close_pairs.head())

# load cv-valid-dev.csv
df = pd.read_csv(f"{DIRECTORY}cv-valid-dev.csv")
print(df.head())

now = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
torch.autograd.set_detect_anomaly(True)
device = "cuda" if torch.cuda.is_available() else "cpu"
default_model = whisper.load_model("tiny", device=device)
# loop through the close_pairs dataset and print the corresponding text from the cv-valid-dev.csv dataset
for index, row in df_close_pairs.iterrows():

    filename1 = df[df["text"] == row["phrase1"]]["filename"].values[0]
    text1 = df[df["text"] == row["phrase1"]]["text"].values[0]
    filename2 = df[df["text"] == row["phrase2"]]["filename"].values[0]
    text2 = df[df["text"] == row["phrase2"]]["text"].values[0]
    id = row["id"]
    if id == 7:
        print(f"{filename1}: {text1}")
        print(f"{filename2}: {text2}")
        print("\n")
        output_file = audio_distance(
            f"{DIRECTORY}{filename1}", f"{DIRECTORY}{filename2}", id=id, now=now
        )

        print(f"whisper transcribe {output_file}")
        # audio, sr = load_audio_file(output_file)
        # print(audio)

        try:
            result = default_model.transcribe(output_file)
            print(result)
        except RuntimeError as e:
            print(e)
