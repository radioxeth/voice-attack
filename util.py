import librosa
import librosa.display
import torchaudio
import torch
import matplotlib.pyplot as plt
import numpy as np


def load_audio_file(file_path):
    y, sr = torchaudio.load(file_path)
    return y, sr


def resample_audio(audio, sr, new_sr):
    resampler = torchaudio.transforms.Resample(sr, new_sr)
    audio_resampled = resampler(audio)
    return audio_resampled, new_sr


def compute_mel_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=80):
    # Function to compute a spectrogram from an audio signal
    # transform = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=sr, n_fft=n_fft, hop_length=hop_length
    # )
    # log_S = transform(audio)
    audio = audio.cpu()  # Move tensor to CPU if it's not already
    audio = audio.numpy()  # Convert to NumPy array
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_S = librosa.power_to_db(S=S, ref=np.max)
    return log_S


## compute the levenshtein distance
def compute_levenshtein_distance(s1, s2):
    return torchaudio.functional.edit_distance(s1, s2)


def compute_euclidean_distances(log_spectrogram1, log_spectrogram2):
    # Subtract the two spectrograms element-wise
    difference = log_spectrogram1 - log_spectrogram2

    # Square the difference
    squared_difference = np.square(difference)

    # compute the Euclidean distance
    euclidean_distances = np.sqrt(squared_difference)

    return euclidean_distances


def audio_mfcc_transform(audio, sr, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=23):
    mfcc_transform = torchaudio.transforms.MFCC(
        n_mfcc=n_mfcc, sample_rate=sr, log_mels=True
    )
    mfcc = mfcc_transform(audio)
    return mfcc


### function to vad the waveform
def vad_waveform(audio, sr):
    # detect voice activity
    audio = torchaudio.functional.vad(
        audio,
        sr,
        trigger_time=0.55,
        trigger_level=7,
        noise_reduction_amount=1.65,
        search_time=0.55,
    )
    return audio


### function to center and pad the waveform
def center_and_pad_waveforms(audio1, sr1, audio2, sr2):
    audio1 = audio1.cpu()  # Move tensor to CPU if it's not already
    audio1 = audio1.numpy()  # Convert to NumPy array

    audio2 = audio2.cpu()  # Move tensor to CPU if it's not already
    audio2 = audio2.numpy()  # Convert to NumPy array
    # Detect the onsets
    onset_frames1 = librosa.onset.onset_detect(y=audio1, sr=sr1)
    onset_frames2 = librosa.onset.onset_detect(y=audio2, sr=sr2)

    # Convert frames to sample indices
    onset_samples1 = librosa.frames_to_samples(onset_frames1)[0]
    onset_samples2 = librosa.frames_to_samples(onset_frames2)[0]

    # Determine the maximum onset sample index to use as a reference point for alignment
    max_onset = max(onset_samples1, onset_samples2)

    # Pad the beginning of each audio signal with zeros to align the onsets
    audio1_padded = np.pad(audio1, (max_onset - onset_samples1, 0), mode="constant")
    audio2_padded = np.pad(audio2, (max_onset - onset_samples2, 0), mode="constant")

    # Trim or extend both signals to the same length
    max_length = max(len(audio1_padded), len(audio2_padded))
    audio1_padded = librosa.util.fix_length(audio1_padded, size=max_length)
    audio2_padded = librosa.util.fix_length(audio2_padded, size=max_length)

    audio1 = torch.from_numpy(audio1_padded)
    audio2 = torch.from_numpy(audio2_padded)
    return audio1, audio2


def print_waveform(
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
    num_frames = waveform.shape[0]  # Updated to handle single channel waveform
    time_axis = torch.arange(0, num_frames) / sr

    # # Create a figure and a single axis
    fig, ax = plt.subplots(figsize=(10, 4))

    # # Plotting the waveform
    ax.plot(time_axis, waveform, linewidth=1)  # Assuming waveform[0] for single channel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)

    # # Set super title if provided
    if suptitle is not None:
        plt.suptitle(suptitle)

    # # Adjust the margins
    fig.subplots_adjust(left=left_margin, right=right_margin)
    plt.tight_layout()
    plt.savefig(file_name)


### Define a function to print the figure
def print_figure(audio_path, figure_title, log_spectrogram, sr):
    # # Visualize the original and equalized Mel spectrograms
    # if isinstance(log_spectrogram, torch.Tensor):
    #     log_spectrogram = log_spectrogram.cpu().numpy()
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 1, 1)
    librosa.display.specshow(log_spectrogram, sr=sr, x_axis="time", y_axis="mel")
    plt.title(figure_title)
    plt.colorbar(format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(audio_path)


# Define a function to print the Euclidean distance figure
def print_euclidean_distance_figure(
    figure_name, figure_title, spectrogram1, spectrogram2, sr, filter_level
):
    # Ensure the spectrograms are the same shape
    if spectrogram1.shape == spectrogram2.shape:
        euclidean_distances = compute_euclidean_distances(spectrogram1, spectrogram2)

        distances_filtered = np.maximum(euclidean_distances, filter_level)
        print_figure(
            figure_name,
            figure_title,
            distances_filtered,
            sr,
        )
    else:
        print(
            "Spectrograms are not of the same shape, can't compute Euclidean distance."
        )


### convert audio_waveform to audio as a .wav file
def save_wav(audio_waveform, sr, wav_file_path, format="wav"):
    # Convert the 1D NumPy array to a 2D array with shape [1, N]
    audio_waveform_2d = np.expand_dims(audio_waveform, axis=0)
    # Convert the NumPy array to a PyTorch tensor
    audio_waveform_tensor = torch.from_numpy(audio_waveform_2d).float()
    # Save the waveform as a .wav file
    torchaudio.save(
        wav_file_path,
        audio_waveform_tensor,
        sr,
        format=format,
    )


def clean_text(text, clean=True, strip=False, lower=False):
    if clean:
        for symbol in "_-!'(),.:;?":
            text = text.replace(symbol, "")
    if strip:
        text = text.strip()
    if lower:
        text = text.lower()
    return text
