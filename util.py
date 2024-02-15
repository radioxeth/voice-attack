import librosa
import librosa.display
import torchaudio
import torch
import matplotlib.pyplot as plt
import numpy as np


def load_audio_file(file_path):
    # Function to load an audio file
    y, sr = librosa.load(file_path, mono=False)
    print(sr)
    return y, sr


def compute_spectrogram(audio, sr):
    # Function to compute a spectrogram from an audio signal
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
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


### function to center and pad the waveform
def center_and_pad_waveforms(y1, sr1, y2, sr2):
    # Detect the onsets
    onset_frames1 = librosa.onset.onset_detect(y=y1, sr=sr1)
    onset_frames2 = librosa.onset.onset_detect(y=y2, sr=sr2)

    # Convert frames to sample indices
    onset_samples1 = librosa.frames_to_samples(onset_frames1)[0]
    onset_samples2 = librosa.frames_to_samples(onset_frames2)[0]

    # Determine the maximum onset sample index to use as a reference point for alignment
    max_onset = max(onset_samples1, onset_samples2)

    # Pad the beginning of each audio signal with zeros to align the onsets
    y1_padded = np.pad(y1, (max_onset - onset_samples1, 0), mode="constant")
    y2_padded = np.pad(y2, (max_onset - onset_samples2, 0), mode="constant")

    # Trim or extend both signals to the same length
    max_length = max(len(y1_padded), len(y2_padded))
    y1_padded = librosa.util.fix_length(y1_padded, size=max_length)
    y2_padded = librosa.util.fix_length(y2_padded, size=max_length)

    return y1_padded, y2_padded


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
    # Visualize the original and equalized Mel spectrograms
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
def save_wav(audio_waveform, sr, wav_file_path):
    # Convert the 1D NumPy array to a 2D array with shape [1, N]
    audio_waveform_2d = np.expand_dims(audio_waveform, axis=0)
    # Convert the NumPy array to a PyTorch tensor
    audio_waveform_tensor = torch.from_numpy(audio_waveform_2d).float()
    # Save the waveform as a .wav file
    torchaudio.save(
        wav_file_path,
        audio_waveform_tensor,
        sr,
        format="wav",
    )
