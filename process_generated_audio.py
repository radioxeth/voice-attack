# open close_pair.csv and see the result
from audio_distance import audio_distance
import pandas as pd
from datetime import datetime
import whisper
from util import load_audio_file
import torch

COMBINED_RESULTS = "generated_results.csv"

# Load the close_pairs dataset
df_close_pairs = pd.read_csv(COMBINED_RESULTS)
print(df_close_pairs.head())


now = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
torch.autograd.set_detect_anomaly(True)
device = "cuda" if torch.cuda.is_available() else "cpu"
default_model = whisper.load_model("tiny", device=device)
# loop through the df_close_pairs dataset for each unique pair_id
previous_row = None
for index, row in df_close_pairs.iterrows():
    if index % 2 == 1:
        if previous_row is not None:
            filename1 = previous_row["filename"]
            filename2 = row["filename"]
            id1 = previous_row["pair_id"]
            id2 = row["pair_id"]
            pair_id = previous_row["pair_id"]
            transcript1 = previous_row["transcript"]
            transcript2 = row["transcript"]
            print(f"{pair_id}.{id1} {filename1}: {transcript1}")
            print(f"{pair_id}.{id2} {filename2}: {transcript2}")
            id = f"{pair_id}-{index}"
            try:
                output_file, mfcc = audio_distance(filename1, filename2, id=id, now=now)
            except RuntimeError as e:
                print(e)

    previous_row = row
    # print(df_close_pairs_pair)
    # for index, row in df_close_pairs_pair.iterrows():
    #     filename1 = row["filename"]
    #     text1 = row["transcript"]
    #     filename2 = row["filename"]
    #     text2 = row["transcript"]
    #     id = row["id"]
    #     print(f"{filename1}: {text1}")
    #     print(f"{filename2}: {text2}")
    #     print("\n")
    # output_file = audio_distance(f"{filename1}", f"{filename2}", id=id, now=now)

# for index, row in df_close_pairs.iterrows():
#     if index % 2 == 0:
#         print(f"{row['filename']}: {row['text']}")
# filename1 = df[df["text"] == row["phrase1"]]["filename"].values[0]
# text1 = df[df["text"] == row["phrase1"]]["text"].values[0]
# filename2 = df[df["text"] == row["phrase2"]]["filename"].values[0]
# text2 = df[df["text"] == row["phrase2"]]["text"].values[0]
# id = row["id"]
# if id == 7:
#     print(f"{filename1}: {text1}")
#     print(f"{filename2}: {text2}")
#     print("\n")
#     output_file = audio_distance(
#         f"{DIRECTORY}{filename1}", f"{DIRECTORY}{filename2}", id=id, now=now
#     )

#     print(f"whisper transcribe {output_file}")
#     # audio, sr = load_audio_file(output_file)
#     # print(audio)

#     try:
#         result = default_model.transcribe(output_file)
#         print(result)
#     except RuntimeError as e:
#         print(e)
