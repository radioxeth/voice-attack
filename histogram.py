import librosa
import librosa.display
from librosa.util import fix_length
import matplotlib.pyplot as plt
from skimage import exposure
import numpy as np


AUDIO_NAME="bonapatite"
print(AUDIO_NAME)
audio_path = f"audio/{AUDIO_NAME}.wav"

# Load and resample the audio file to 8192 Hz
y, sr = librosa.load(audio_path)
print(f"sample rate = {sr}")
# Calculate the number of samples for 3 seconds
target_length = 3 * sr

# Trim or pad the signal to have a length of 3 seconds (sr * 3 samples)
y = fix_length(y, size=target_length)

# Generate a Mel spectrogram with the new sample rate and desired n_mels
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# Convert the power spectrogram to dB
log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

# Apply histogram equalization
# Note: The `exposure.equalize_hist` function expects a 2D array with values in the range [0, 1].
# You may need to normalize log_spectrogram before equalization and then scale it back after.
normalized_log_spectrogram = (log_spectrogram - log_spectrogram.min()) / (log_spectrogram.max() - log_spectrogram.min())
equalized_log_spectrogram = exposure.equalize_hist(normalized_log_spectrogram)

# Scale the equalized spectrogram back to the original dB range
equalized_log_spectrogram = equalized_log_spectrogram * (log_spectrogram.max() - log_spectrogram.min()) + log_spectrogram.min()

# Visualize the original and equalized Mel spectrograms
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
librosa.display.specshow(log_spectrogram, sr=sr, x_axis="time", y_axis="mel")
plt.title("Original Mel spectrogram")
plt.colorbar(format="%+2.0f dB")

plt.subplot(2, 1, 2)
librosa.display.specshow(equalized_log_spectrogram, sr=sr, x_axis="time", y_axis="mel")
plt.title("Histogram Equalized Mel spectrogram")
plt.colorbar(format="%+2.0f dB")

plt.tight_layout()
plt.savefig(f"{AUDIO_NAME}.png")
