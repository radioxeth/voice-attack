import librosa
import librosa.display
import numpy as np
import torchaudio
import torch


AUDIO_NAME1 = "bonapatite"
print(AUDIO_NAME1)
audio_path1 = f"audio/{AUDIO_NAME1}.wav"

AUDIO_NAME2 = "bonappletea"
print(AUDIO_NAME2)
audio_path2 = f"audio/{AUDIO_NAME2}.wav"

# Load the audio files
y1, sr1 = librosa.load("audio/bonappletea.wav", sr=8192, mono=False)
y2, sr2 = librosa.load("audio/bonapatite.wav", sr=8192, mono=False)


waveform_euclidean_distances = compute_euclidean_distances(y1_padded, y2_padded)

## adjust the amplitudes of the waveform to be between -1 and 1
waveform_euclidean_distances_dif = waveform_euclidean_distances - (
    waveform_euclidean_distances / 2
)

print_waveform(
    waveform_euclidean_distances_dif,
    sr1,
    title="Waveform Euclidean Distances",
    file_name="images/distances_waveform_dif.png",
)

# Convert the 1D NumPy array to a 2D array with shape [1, N]
audio_waveform_2d = np.expand_dims(waveform_euclidean_distances_dif, axis=0)

# Convert the NumPy array to a PyTorch tensor
audio_waveform_tensor = torch.from_numpy(audio_waveform_2d).float()

torchaudio.save(
    "audio/distances_waveform_dif.wav", audio_waveform_tensor, sr1, format="wav"
)

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


print_waveform(y1_padded, sr1, title="Waveform 1", file_name="images/waveform1.png")
print_waveform(y2_padded, sr2, title="Waveform 2", file_name="images/waveform2.png")

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
# euclidean_distances = compute_euclidean_distances(log_spectrogram1, log_spectrogram2)
# euclidean_distances = np.maximum(euclidean_distances, 0.1)
# print("type of euclidean_distances", type(euclidean_distances))
# print("shape", euclidean_distances.shape)

# print("type of spectrogram", type(log_spectrogram1))
# print("shape", log_spectrogram1.shape)

# audio_waveform = librosa.feature.inverse.mel_to_audio(euclidean_distances)

# print_waveform(
#     audio_waveform,
#     sr1,
#     title="Waveform Euclidean Distances",
#     file_name="images/distances_waveform.png",
# )

# # Convert the 1D NumPy array to a 2D array with shape [1, N]
# audio_waveform_2d = np.expand_dims(audio_waveform, axis=0)

# # Convert the NumPy array to a PyTorch tensor
# audio_waveform_tensor = torch.from_numpy(audio_waveform_2d).float()


# ### convert audio_waveform to audio as a .wav file
# torchaudio.save(
#     "audio/distances_waveform.wav",
#     audio_waveform_tensor,
#     sr1,
#     format="wav",
# )


# # Convert the 1D NumPy array to a 2D array with shape [1, N]
# audio_waveform1_2d = np.expand_dims(y1_padded, axis=0)

# # Convert the NumPy array to a PyTorch tensor
# audio_waveform1_tensor = torch.from_numpy(audio_waveform_2d).float()

# ### convert audio_waveform to audio as a .wav file
# torchaudio.save(
#     "audio/waveform1.wav",
#     audio_waveform1_tensor,
#     sr1,
#     format="wav",
# )
