import torch
import torchaudio
import csv
from util import save_wav
from util import clean_text
from datetime import datetime
import os
import whisper
from Levenshtein import distance
from util import texts
from util import paired_texts
import pandas as pd

cv = pd.read_csv("cv-valid-dev/unique-text-rows.csv")
texts = cv["text"].tolist()
texts = texts[329:]
# texts = texts[631:892]

paired_texts = [{"id": i + 1, "text": texts[i]} for i in range(len(texts))]


torch.autograd.set_detect_anomaly(True)
device = "cuda" if torch.cuda.is_available() else "cpu"
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

model = whisper.load_model("tiny", device=device)

now = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
tts_dir = f"audio_generated/tts{now}"
os.makedirs(f"{tts_dir}", exist_ok=True)

csv_filename = f"{tts_dir}/results.csv"
file_exists = os.path.isfile(csv_filename)

with open(csv_filename, "a") as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["id", "text", "transcript", "filename", "tts_distance"])

    for item in paired_texts:
        try:
            text = item["text"]
            id = item["id"]
            # pair_id = item["pair_id"]
            with torch.inference_mode():
                # generate waveform from text
                processed, lengths = processor(text)
                processed = processed.to(device)
                lengths = lengths.to(device)
                spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
                waveforms, lengths = vocoder(spec, spec_lengths)
                sample_rate = vocoder.sample_rate
                if device == "cuda":
                    waveforms = waveforms.cpu()

                text = clean_text(text, clean=True, lower=True, strip=True)
                wav_filename = f"{tts_dir}/{text}.wav"
                save_wav(waveforms[0], sample_rate, wav_filename)
                print(f"Saved {wav_filename}")

                # transcribe whisper
                print("whisper transcribe")
                result = model.transcribe(wav_filename, fp16=False)
                print(result["text"])
                transcript = result["text"]
                transcript = clean_text(transcript, clean=True, lower=True, strip=True)

                # calculate distance
                tts_distance = distance(text, transcript)
                writer.writerow([id, text, transcript, wav_filename, tts_distance])
        except:
            print(f"error tts {id} {text}")
            continue
