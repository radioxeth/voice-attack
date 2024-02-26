import torch
import torchaudio
import csv
from util import save_wav
from util import clean_text
from datetime import datetime
import os
import whisper
from Levenshtein import distance

texts = [
    "For all intensive purposes.",
    "For all intents and purposes.",
    "Nip it in the butt.",
    "Nip it in the bud.",
    "Self-depreciating.",
    "Self-deprecating.",
    "Escape goat.",
    "Scapegoat.",
    "Mute point.",
    "Moot point.",
    "Old-timer's disease.",
    "Alzheimer's disease.",
    "Doggy-dog world.",
    "Dog-eat-dog world.",
    "Lack toast and tolerant.",
    "Lactose intolerant.",
    "Bowl in a china shop.",
    "Bull in a china shop.",
    "Deep-seeded.",
    "Deep-seated.",
    "Taken for granite.",
    "Taken for granted.",
    "Case and point.",
    "Case in point.",
    "An escape goat.",
    "A scapegoat.",
    "Pass mustard.",
    "Pass muster.",
    "On tender hooks.",
    "On tenterhooks.",
    "Tongue and cheek.",
    "Tongue in cheek.",
    "Card shark.",
    "Card sharp.",
    "Damp squid.",
    "Damp squib.",
    "Curl up in the feeble position.",
    "Curl up in the fetal position.",
    "A hard road to hoe.",
    "A hard row to hoe.",
    "Ex-patriot.",
    "Expatriate.",
    "Biting my time.",
    "Biding my time.",
    "Antidotal evidence.",
    "Anecdotal evidence.",
    "Circus-sized.",
    "Circumcised.",
    "Hunger pains.",
    "Hunger pangs.",
    "Flush out the details.",
    "Flesh out the details.",
    "He's a wolf in cheap clothing.",
    "He's a wolf in sheep's clothing.",
    "Pre-Madonna.",
    "Prima donna.",
    "Social leper.",
    "Social pariah.",
    "Give free rein.",
    "Give free reign.",
    "Make ends meat.",
    "Make ends meet.",
    "Right from the gecko.",
    "Right from the get-go.",
    "Stock home syndrome.",
    "Stockholm syndrome.",
    "Chester drawers.",
    "Chest of drawers.",
    "Beckon call.",
    "Beck and call.",
    "Full-proof.",
    "Foolproof.",
    "Two peas in a pot.",
    "Two peas in a pod.",
    "On the spurt of the moment.",
    "On the spur of the moment.",
    "Mind-bottling.",
    "Mind-boggling.",
    "I plead the Fifth Commandment.",
    "I plead the Fifth Amendment.",
]
paired_texts = [
    {"id": i + 1, "pair_id": (i // 2) + 1, "text": texts[i]} for i in range(len(texts))
]


torch.autograd.set_detect_anomaly(True)
device = "cuda" if torch.cuda.is_available() else "cpu"
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

model = whisper.load_model("tiny", device=device)

now = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
tts_dir = f"generated_audio/tts{now}"
os.makedirs(f"{tts_dir}", exist_ok=True)

csv_filename = f"{tts_dir}/results.csv"
file_exists = os.path.isfile(csv_filename)

with open(csv_filename, "a") as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(
            ["id", "pair_id", "text", "transcript", "filename", "tts_distance"]
        )

    for item in paired_texts:
        text = item["text"]
        id = item["id"]
        pair_id = item["pair_id"]
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
            writer.writerow([id, pair_id, text, transcript, wav_filename, tts_distance])
