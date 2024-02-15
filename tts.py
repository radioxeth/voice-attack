import torch
import torchaudio
import IPython
import matplotlib.pyplot as plt
from util import save_wav
from datetime import datetime
import os

now = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

# torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

# print(torch.__version__)
# print(torchaudio.__version__)
# print(device)

# symbols = "_-!'(),.:;? abcdefghijklmnopqrstuvwxyz"
# look_up = {s: i for i, s in enumerate(symbols)}
# symbols = set(symbols)


# def text_to_sequence(text):
#     text = text.lower()
#     return [look_up[s] for s in text if s in symbols]


bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)


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

tts_dir = f"generated_audio/tts{now}"
os.makedirs(f"{tts_dir}", exist_ok=True)
for text in texts:

    with torch.inference_mode():
        processed, lengths = processor(text)
        processed = processed.to(device)
        lengths = lengths.to(device)
        spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
        waveforms, lengths = vocoder(spec, spec_lengths)
        sample_rate = vocoder.sample_rate
        if device == "cuda":
            waveforms = waveforms.cpu()
        ## remove "_-!'(),.:;? from text
        for symbol in "_-!'(),.:;?":
            text = text.replace(symbol, "")
        save_wav(waveforms[0], sample_rate, f"{tts_dir}/{text}.wav")
