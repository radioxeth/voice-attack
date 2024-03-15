import librosa
import librosa.display
import torchaudio
import torch
import matplotlib.pyplot as plt
import numpy as np

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


CLASS_RECORDED = 1
CLASS_TTS = -1


def load_audio_file(file_path):
    y, sr = torchaudio.load(file_path)
    return y, sr


def resample_audio(audio, sr, new_sr):
    resampler = torchaudio.transforms.Resample(sr, new_sr)
    audio_resampled = resampler(audio)
    return audio_resampled, new_sr


def compute_mel_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=80):
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


def normalize_waveform(audio):
    # find max and normalize audio to 1
    audio_max = np.abs(audio).max()

    # normalize audio
    audio = audio / audio_max
    return audio


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
import numpy as np
import librosa
import torch


def center_and_pad_waveforms(audio1, sr1, audio2, sr2):
    # Assuming audio1 and audio2 are PyTorch tensors
    audio1 = audio1.cpu().numpy()  # Convert to NumPy array
    audio2 = audio2.cpu().numpy()  # Convert to NumPy array

    # Detect the onsets
    onset_frames1 = librosa.onset.onset_detect(y=audio1, sr=sr1)
    onset_frames2 = librosa.onset.onset_detect(y=audio2, sr=sr2)

    # Convert frames to sample indices, safely handle empty cases
    onset_samples1 = (
        librosa.frames_to_samples(onset_frames1)[0] if onset_frames1.size > 0 else 0
    )
    onset_samples2 = (
        librosa.frames_to_samples(onset_frames2)[0] if onset_frames2.size > 0 else 0
    )

    if onset_frames1.size > 0:
        onset_samples1 = librosa.frames_to_samples(onset_frames1)[0]
    else:
        onset_samples1 = 0  # Fallback value if no onsets are detected

    if onset_frames2.size > 0:
        onset_samples2 = librosa.frames_to_samples(onset_frames2)[0]
    else:
        onset_samples2 = 0  # Fallback value if no onsets are detected

    # Determine the maximum onset sample index to use as a reference point for alignment
    max_onset = max(onset_samples1, onset_samples2)

    # Pad the beginning of each audio signal with zeros to align the onsets
    audio1_padded = np.pad(audio1, (max_onset - onset_samples1, 0), mode="constant")
    audio2_padded = np.pad(audio2, (max_onset - onset_samples2, 0), mode="constant")

    # Trim or extend both signals to the same length
    max_length = max(len(audio1_padded), len(audio2_padded))
    audio1_padded = librosa.util.fix_length(audio1_padded, size=max_length)
    audio2_padded = librosa.util.fix_length(audio2_padded, size=max_length)

    # Convert back to tensors
    audio1 = torch.from_numpy(audio1_padded)
    audio2 = torch.from_numpy(audio2_padded)

    return audio1, audio2


# def center_and_pad_waveforms(audio1, sr1, audio2, sr2):
#     audio1 = audio1.cpu()  # Move tensor to CPU if it's not already
#     audio1 = audio1.numpy()  # Convert to NumPy array

#     audio2 = audio2.cpu()  # Move tensor to CPU if it's not already
#     audio2 = audio2.numpy()  # Convert to NumPy array
#     # Detect the onsets
#     onset_frames1 = librosa.onset.onset_detect(y=audio1, sr=sr1)
#     onset_frames2 = librosa.onset.onset_detect(y=audio2, sr=sr2)

#     # Convert frames to sample indices
#     onset_samples1 = librosa.frames_to_samples(onset_frames1)[0]
#     onset_samples2 = librosa.frames_to_samples(onset_frames2)[0]

#     # Determine the maximum onset sample index to use as a reference point for alignment
#     max_onset = max(onset_samples1, onset_samples2)

#     # Pad the beginning of each audio signal with zeros to align the onsets
#     audio1_padded = np.pad(audio1, (max_onset - onset_samples1, 0), mode="constant")
#     audio2_padded = np.pad(audio2, (max_onset - onset_samples2, 0), mode="constant")

#     # Trim or extend both signals to the same length
#     max_length = max(len(audio1_padded), len(audio2_padded))
#     audio1_padded = librosa.util.fix_length(audio1_padded, size=max_length)
#     audio2_padded = librosa.util.fix_length(audio2_padded, size=max_length)

#     audio1 = torch.from_numpy(audio1_padded)
#     audio2 = torch.from_numpy(audio2_padded)
#     return audio1, audio2


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
    plt.close()


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
    plt.close()


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


def mfccs_class_ids_from_files(directory_path, files, mfccs, class_ids, class_id):
    for file in files:
        mfcc = np.load(f"{directory_path}/{file}")
        mfccs.append(mfcc)
        class_ids.append(class_id)
    return mfccs, class_ids


def shuffle_and_split(files, split_ratio=0.8):
    # select from recorded_files into two random sets, one for training and one for inference
    np.random.shuffle(files)

    # split recorded_files into two arrays
    files_train = files[: int(split_ratio * len(files))]
    files_test = files[int((split_ratio) * len(files)) :]
    print(f"Train: {len(files_train)} Test: {len(files_test)}")
    return files_train, files_test
