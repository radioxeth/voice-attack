# open audio_generated/tts2024-03-13--08-38-57_good/results.csv as csv
#
# for each row in csv:
#     if row["tts_distance"] < 10:
#         copy row["filename"] to audio_generated/tts2024-03-13--08-38-57_good
#     else:
#         copy row["filename"] to audio_generated/tts2024-03-13--08-38-57_bad

import pandas as pd
import os

# Step 1: Load the CSV file
# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv("audio_generated/tts2024-03-12--14-43-46_good/results.csv")

# add 2081 to each id
df["id"] = df["id"] + 2081

# save to new csv
df.to_csv("audio_generated/good1.csv", index=False)
