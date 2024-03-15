# open close_pair.csv and see the result
from audio_distance import audio_distance
import pandas as pd
from datetime import datetime
import whisper
from util import load_audio_file
import torch
import os
import csv
from util import texts
from util import paired_texts

torch.autograd.set_detect_anomaly(True)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
default_model = whisper.load_model("tiny", device=device)


# write to csv with headers of path, filename, transcript
def transcribe_euc_distance(directory_path):
    # get files from directory with eud_distance in the name

    # List all files and directories in the specified path
    entries = os.listdir(directory_path)

    # If you want to list files only
    files = [
        entry
        for entry in entries
        if os.path.isfile(os.path.join(directory_path, entry))
        and "euc_distance" in entry
    ]
    # print(files)
    # open .npy files and get the mfcc

    csv_filename = f"{directory_path}/generated_lev_distance.csv"
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, "a") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["pair_id", "filename", "text"])
        for file in files:
            try:
                result = default_model.transcribe(
                    os.path.join(directory_path, file), fp16=False
                )
                print(f"filename: {file}")
                print(result["text"])
                pair_id = file.split("-")[0]
                writer.writerow([pair_id, file, result["text"]])
            except RuntimeError as e:
                print(e)


transcribe_euc_distance("generated_audio/2024-03-02--15-25-46")
