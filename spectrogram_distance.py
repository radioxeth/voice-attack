import librosa
import librosa.display
from librosa.util import fix_length
import matplotlib.pyplot as plt
from skimage import exposure
import numpy as np
import torchaudio
import torch


def compute_minkowski_distances(log_spectrogram1, log_spectrogram2, p=2):
    # Compute the absolute difference to the power of p
    difference = np.abs(log_spectrogram1 - log_spectrogram2) ** p

    # Sum the differences and take the p-th root to compute the Minkowski distance
    # Since your operation seems to be element-wise, you'll need to adjust how you compute the root.
    # For a true Minkowski distance across all elements, you would sum up all elements and then take the p-th root.
    # Here's an element-wise approach, which differs conceptually from the standard Minkowski distance calculation.
    minkowski_distances = difference ** (1 / p)

    return minkowski_distances


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
    print("waveform", waveform)
    print("sr", sr)
    print("shape", waveform.shape)
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

    # # Output file name
    print(f"WAVEFORM -- {file_name}")
    plt.savefig(file_name)


### Define a function to print the figure
def print_figure(audio_name, figure_title, log_spectrogram, sr):
    # Visualize the original and equalized Mel spectrograms
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 1, 1)
    librosa.display.specshow(log_spectrogram, sr=sr, x_axis="time", y_axis="mel")
    plt.title(figure_title)
    plt.colorbar(format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(f"{audio_name}.png")


def compute_euclidean_distances(log_spectrogram1, log_spectrogram2):
    # Subtract the two spectrograms element-wise
    difference = log_spectrogram1 - log_spectrogram2

    # Square the difference
    squared_difference = np.square(difference)

    # compute the Euclidean distance
    euclidean_distances = np.sqrt(squared_difference)

    return euclidean_distances


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


# Define a function to print the Minkowski distance figure
def print_minkowski_distance_figure(
    figure_name, figure_title, spectrogram1, spectrogram2, sr, filter_level, p=2
):
    # Ensure the spectrograms are the same shape
    if spectrogram1.shape == spectrogram2.shape:
        minkowski_distances = compute_minkowski_distances(
            spectrogram1, spectrogram2, p=p
        )

        # Filter the distances (this step may need to be adjusted based on your actual requirements)
        distances_filtered = np.maximum(
            minkowski_distances, filter_level
        )  # This line might not make sense in this context

        print_figure(
            figure_name,
            figure_title,
            distances_filtered,
            sr,
        )
    else:
        print(
            "Spectrograms are not of the same shape, can't compute Minkowski distance."
        )


AUDIO_NAME1 = "bonapatite"
print(AUDIO_NAME1)
audio_path1 = f"audio/{AUDIO_NAME1}.wav"

AUDIO_NAME2 = "bonappletea"
print(AUDIO_NAME2)
audio_path2 = f"audio/{AUDIO_NAME2}.wav"

# Load the audio files
y1, sr1 = librosa.load("audio/bonappletea.wav", sr=8192, mono=False)
y2, sr2 = librosa.load("audio/bonapatite.wav", sr=8192, mono=False)


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

print_waveform(y1_padded, sr1, title="Waveform 1", file_name="images/waveform1.png")
print_waveform(y2_padded, sr2, title="Waveform 2", file_name="images/waveform2.png")

### convert audio_waveform to audio as a .wav file
# torchaudio.save(
#     "audio/y1.wav",
#     y1_padded,
#     sr1,
#     format="wav",
# )

# Generate the Mel spectrograms for the padded signals
spectrogram1 = librosa.feature.melspectrogram(y=y1_padded, sr=sr1)
spectrogram2 = librosa.feature.melspectrogram(y=y2_padded, sr=sr2)

# Convert to dB
log_spectrogram1 = librosa.power_to_db(spectrogram1, ref=np.max)
log_spectrogram2 = librosa.power_to_db(spectrogram2, ref=np.max)

equalized_log_spectrogram1 = exposure.equalize_hist(log_spectrogram1)

equalized_log_spectrogram2 = exposure.equalize_hist(log_spectrogram2)

# # # Visualize the original and equalized Mel spectrograms
# print("MFCC 1", librosa.feature.mfcc(y=y1_padded, sr=sr1))
# print("MFCC 2", librosa.feature.mfcc(y=y2_padded, sr=sr2))

# # # Visualize the original and equalized Mel spectrograms
print_figure(
    AUDIO_NAME1, "Histogram Equalized Mel Spectrogram", equalized_log_spectrogram1, sr1
)

print_figure(
    AUDIO_NAME2, "Histogram Equalized Mel Spectrogram", equalized_log_spectrogram2, sr2
)


# print_euclidean_distance_figure(
#     "distance_original",
#     "Euclidean Distance of Mel Spectrograms",
#     log_spectrogram1,
#     log_spectrogram2,
#     sr1,
#     filter_level=0,
# )

# print_euclidean_distance_figure(
#     "distance_equalized",
#     "Euclidean Distance of Histogram Equalized Mel Spectrograms",
#     equalized_log_spectrogram1,
#     equalized_log_spectrogram2,
#     sr1,
#     filter_level=0,
# )

# print_minkowski_distance_figure(
#     "minkowski_distance_original",
#     "Minkowski Distance of Mel Spectrograms",
#     log_spectrogram1,
#     log_spectrogram2,
#     sr1,
#     filter_level=10,
#     p=8,
# )

# print_minkowski_distance_figure(
#     "minkowski_distance_equalized",
#     "Minkowski Distance of Histogram Equalized Mel Spectrograms",
#     equalized_log_spectrogram1,
#     equalized_log_spectrogram2,
#     sr1,
#     filter_level=0,
#     p=4,
# )

# # # Compute the Euclidean distance between the two spectrograms
euclidean_distances = compute_euclidean_distances(log_spectrogram1, log_spectrogram2)
euclidean_distances = np.maximum(euclidean_distances, 0.1)
print("type of euclidean_distances", type(euclidean_distances))
print("shape", euclidean_distances.shape)

print("type of spectrogram", type(log_spectrogram1))
print("shape", log_spectrogram1.shape)

audio_waveform = librosa.feature.inverse.mel_to_audio(euclidean_distances)

# print_waveform(
#     audio_waveform,
#     sr1,
#     title="Waveform Euclidean Distances",
#     file_name="images/distances_waveform.png",
# )

# Convert the 1D NumPy array to a 2D array with shape [1, N]
audio_waveform_2d = np.expand_dims(audio_waveform, axis=0)

# Convert the NumPy array to a PyTorch tensor
audio_waveform_tensor = torch.from_numpy(audio_waveform_2d).float()


### convert audio_waveform to audio as a .wav file
torchaudio.save(
    "audio/distances_waveform.wav",
    audio_waveform_tensor,
    sr1,
    format="wav",
)


# Convert the 1D NumPy array to a 2D array with shape [1, N]
audio_waveform1_2d = np.expand_dims(y1_padded, axis=0)

# Convert the NumPy array to a PyTorch tensor
audio_waveform1_tensor = torch.from_numpy(audio_waveform_2d).float()

### convert audio_waveform to audio as a .wav file
torchaudio.save(
    "audio/waveform1.wav",
    audio_waveform1_tensor,
    sr1,
    format="wav",
)
