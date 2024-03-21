# open close_pair.csv and see the result
from audio_distance import audio_distance
import pandas as pd
from datetime import datetime
import whisper
from util import find_max_loudness, load_audio_file
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


df1 = pd.read_csv(f"{DIRECTORY}/{FILENAME}")
print(df1.head())


#### Recorded Audio Params
DIRECTORY = "cv-valid-dev"
FILENAME = "cv-valid-dev-wav.csv"
class_type_id = CLASS_RECORDED
class_type = "recorded"

df2 = pd.read_csv(f"{DIRECTORY}/{FILENAME}")
print(df2.head())

# concat the filenames
filename1 = df1["filename"]
filename2 = df2["filename"]
filenames = pd.concat([filename1, filename2], ignore_index=True)


max_loudness = find_max_loudness(filenames)
print(max_loudness)
