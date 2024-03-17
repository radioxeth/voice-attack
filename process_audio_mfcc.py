# open close_pair.csv and see the result
from audio_distance import audio_distance
import pandas as pd
from datetime import datetime
from util import CLASS_RECORDED
from util import CLASS_TTS
from audio_mfcc import audio_mfcc
import os
import csv

MAX_AMPLITUDE = 0.9999695

#### Generated Audio Params
DIRECTORY_GEN = "audio_generated/tts_final"
FILENAME_GEN = "results.csv"
class_type_id_GEN = CLASS_TTS
class_type_GEN = "generated"

#### Recorded Audio Params
DIRECTORY_REC = "cv-valid-dev"
FILENAME_REC = "cv-valid-dev-wav.csv"
class_type_id_REC = CLASS_RECORDED
class_type_REC = "recorded"

class_definitions = [
    (DIRECTORY_GEN, FILENAME_GEN, class_type_GEN, class_type_id_GEN),
    (DIRECTORY_REC, FILENAME_REC, class_type_REC, class_type_id_REC),
]

# loop through nmels range 1:40
for n_mels in range(1, 41):
    # loop through recorded and generated audio
    for DIRECTORY, FILENAME, class_type, class_type_id in class_definitions:

        df = pd.read_csv(f"{DIRECTORY}/{FILENAME}")
        print(df.head())

        # now = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        now = f"final_{n_mels}"

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
                f"{filename}",
                id=index,
                now=now,
                class_type=class_type,
                print_figures=0,
                max_loudness=MAX_AMPLITUDE,
                n_mels=n_mels,
            )
            # write to csv
            with open(csv_filename, "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([index, f"{output_dir}/{output_file}", class_type_id])
                csvfile.close()
