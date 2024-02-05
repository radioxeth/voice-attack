import os
import torch
import librosa
import numpy as np
import pandas as pd
import librosa.display
from AudioUtil import AudioUtil
import matplotlib.pyplot as plt
from IPython.display import Audio
import torchaudio.functional as F
import torchaudio.transforms as T
from scipy.io.wavfile import write
from sklearn import preprocessing
from matplotlib import interactive
from matplotlib.patches import Rectangle
import pandas as pd
import os

run_waveform = False


def plot_waveform(
    waveform,
    sr,
    title="Waveform",
    file_name="images/waveform.png",
    suptitle=None,
    ylabel="Amplitude",
    xlabel="Time (s)",
    left_margin=0.075,  # Default value for left margin
    right_margin=0.95,  # Default value for right margin
):
    waveform = waveform.numpy()
    num_frames = waveform.shape[1]  # Updated to handle single channel waveform
    time_axis = torch.arange(0, num_frames) / sr

    # Create a figure and a single axis
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plotting the waveform
    ax.plot(
        time_axis, waveform[0], linewidth=1
    )  # Assuming waveform[0] for single channel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)

    # Set super title if provided
    if suptitle is not None:
        plt.suptitle(suptitle)

    # Adjust the margins
    fig.subplots_adjust(left=left_margin, right=right_margin)

    # Output file name
    print(f"WAVEFORM -- {file_name}")

    return plt


import matplotlib.pyplot as plt
import librosa


def plot_spectrogram(
    specgram,
    title=None,
    ylabel="Frequency Bin (Hz)",
    xlabel="frame",
    file_name="spectrogram.png",
    suptitle=None,
    left_margin=0.075,
    right_margin=0.95,
    cmap="viridis",
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))  # Create 2 subplots for 2 channels

    # for i, ax in enumerate(axs):
    i = 0
    if title is not None:
        ax.set_title(f"{title} - Channel {i+1}")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    S_dB = librosa.power_to_db(specgram[i], ref=np.max)
    ax.imshow(
        specgram[i],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    img = ax.imshow(
        S_dB, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap
    )

    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    if suptitle is not None:
        plt.suptitle(suptitle)

    fig.subplots_adjust(
        left=left_margin, right=right_margin, hspace=0.3
    )  # Adjust spacing

    print(f"SPECTROGRAM -- {file_name}")

    return plt


######################### THE CODE BELOW SAVE AUDIO AS IMAGE FOR FEATURE INSPECTION ###########################################


# Read metadata file"
metadata_file = f"metadata.csv"
df = pd.read_csv(metadata_file)
df.head()

# Take relevant columns
df = df[["relative_path", "class_id"]]
df.head()


current_directory = os.getcwd()
IMAGE_DIR = f"images2"
WAVEFORM_DIR = "waveforms"

os.makedirs(f"{IMAGE_DIR}", exist_ok=True)
os.makedirs(f"{WAVEFORM_DIR}", exist_ok=True)

for index in range(len(df)):
    try:
        AUDIO_FILE = f'{current_directory}/{df.loc[index, "relative_path"]}'
        CLASS_ID = df.loc[index, "class_id"]
        audio = AudioUtil.open(AUDIO_FILE)
        sig, sr = audio
        sample_rate = 32000
        duration = 4000
        channel = 1
        shift_pct = 0
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        resample_audio = AudioUtil.resample(audio, sample_rate)
        rechannel_audio = AudioUtil.rechannel(resample_audio, channel)
        # noise_aud = AudioUtil.add_noise(rechan, 0.05)
        # dur_aud = AudioUtil.pad_trunc(rechan, duration)
        # shift_aud = AudioUtil.time_shift(dur_aud, shift_pct)

        spectro_gram = AudioUtil.spectro_gram(
            rechannel_audio, n_mels=32, n_fft=1024, hop_len=None, top_db=29
        )

        print(spectro_gram)

        # aug_sgram = AudioUtil.spectro_augment(
        #     sgram, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1
        # )
        file_name = f'{df.loc[index, "relative_path"]}.png'
        base_name = os.path.basename(file_name)

        spectrogram_file_name = f"{IMAGE_DIR}/{base_name}"

        spectrogram = plot_spectrogram(
            spectro_gram,
            file_name=spectrogram_file_name,
            suptitle=f"Mel Spectrogram: {base_name}",
            title=f"Class ID: {CLASS_ID}",
        )
        spectrogram.savefig(spectrogram_file_name)
        spectrogram.close()

        if run_waveform is True:
            waveform_file_name = f"{WAVEFORM_DIR}/{base_name}"
            waveform = plot_waveform(
                sig,
                sr=sr,
                file_name=waveform_file_name,
                suptitle=f"Waveform: {base_name}",
                title=f"Class ID: {CLASS_ID}",
            )
            waveform.savefig(waveform_file_name)
            waveform.close()
    except Exception as e:
        print(f"An error occurred: {e}")
