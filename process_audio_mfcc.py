# open close_pair.csv and see the result
from audio_distance import audio_distance
import pandas as pd
from datetime import datetime
import whisper
from util import load_audio_file
from util import CLASS_RECORDED
from util import CLASS_TTS
from audio_mfcc import audio_mfcc
import torch
import os
import csv


#### Generated Audio Params
DIRECTORY = "audio_generated/tts_final"
FILENAME = "results.csv"
class_type_id = CLASS_TTS
class_type = "generated"

#### Recorded Audio Params
# DIRECTORY = "cv-valid-dev"
# FILENAME = "cv-valid-dev-wav.csv"
# class_type_id = CLASS_RECORDED
# class_type = "recorded"

df = pd.read_csv(f"{DIRECTORY}/{FILENAME}")
print(df.head())
# torch.autograd.set_detect_anomaly(True)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# default_model = whisper.load_model("tiny", device=device)
# loop through the close_pairs dataset and print the corresponding text from the cv-valid-dev.csv dataset
# write output to csv file

now = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
OUTPUT_FILE_DIR = f"mfcc_{class_type}/{now}"

csv_filename = f"{OUTPUT_FILE_DIR}/results.csv"
os.makedirs(f"{OUTPUT_FILE_DIR}", exist_ok=True)

file_exists = os.path.isfile(csv_filename)
with open(csv_filename, "a") as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["id", "filename", "class"])
for index, row in df.iterrows():
    filename = row["filename"]
    mfcc, output_file, output_dir = audio_mfcc(
        f"{filename}", id=index, now=now, class_type=class_type
    )
    # write to csv
    with open(csv_filename, "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([index, f"{output_dir}/{output_file}", class_type_id])
        csvfile.close()
