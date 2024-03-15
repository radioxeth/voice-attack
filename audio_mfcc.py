import os
import argparse
import numpy as np
import torchaudio
import librosa
import torch
from datetime import datetime
from util import (
    CLASS_RECORDED,
    load_audio_file,
    compute_mel_spectrogram,
    compute_euclidean_distances,
    center_and_pad_waveforms,
    print_euclidean_distance_figure,
    save_wav,
    print_waveform,
    print_figure,
    vad_waveform,
    audio_mfcc_transform,
    resample_audio,
    normalize_waveform,
)


def audio_mfcc(
    audio_file1,
    id=0,
    now=datetime.now().strftime("%Y-%m-%d--%H-%M-%S"),
    class_type="recorded",
):

    images_dir = f"images_{class_type}/{now}"
    os.makedirs(f"{images_dir}", exist_ok=True)

    waveforms_dir = f"waveforms_{class_type}/{now}"
    os.makedirs(f"{waveforms_dir}", exist_ok=True)

    mfcc_dir = f"mfcc_{class_type}/{now}"
    os.makedirs(f"{mfcc_dir}", exist_ok=True)

    # Load audio files
    audio1, sr1 = load_audio_file(audio_file1)

    # print_waveform(
    #     audio1,
    #     sr1,
    #     f"Waveform ({class_type})\n{audio_file1}",
    #     f"{waveforms_dir}/{id}-waveform.png",
    # )

    # Resample audio files
    audio1, sr1 = resample_audio(audio1, sr1, 16000)

    # pad audio files to be 6 seconds long
    # audio1 = np.pad(audio1, (0, 32000 - len(audio1)), "constant")

    # print_waveform(audio1, sr1, "Waveform 1", f"waveforms/waveform1 {now}.png")
    audio1 = vad_waveform(audio1[0], sr1)

    # normalize audio
    audio1 = normalize_waveform(audio1)

    # print_waveform(
    #     audio1,
    #     sr1,
    #     f"Processed Waveform ({class_type})\n{audio_file1}",
    #     f"{waveforms_dir}/{id}-processed-waveform.png",
    # )

    # Compute spectrograms
    # spectrogram1 = compute_mel_spectrogram(audio1, sr1, n_mels=13)

    # print_figure(
    #     f"{images_dir}/{id}-mel-spectrogram.png",
    #     f"Mel Spectrogram ({class_type})\n{audio_file1}",
    #     spectrogram1,
    #     sr1,
    # )

    audio_mfcc = audio_mfcc_transform(audio1, sr1, n_mels=13)
    audio_mfcc_filename = f"{id}-{class_type}-mfcc.npy"
    np.save(f"{mfcc_dir}/{audio_mfcc_filename}", audio_mfcc)

    print(f"{class_type} MFCC computed and saved to {audio_mfcc_filename}")
    return audio_mfcc, audio_mfcc_filename, mfcc_dir


def main():

    parser = argparse.ArgumentParser(
        description="Compute distance between two audio files."
    )
    parser.add_argument("audio_file1", type=str, help="Path to the first audio file")

    args = parser.parse_args()
    print("args", args)
    audio_mfcc(args.audio_file1)


if __name__ == "__main__":
    main()
