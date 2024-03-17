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
    print_figures=0,
    max_loudness=None,
    n_mels=13,
):

    mfcc_dir = f"mfcc_{class_type}/{now}"
    os.makedirs(f"{mfcc_dir}", exist_ok=True)

    if print_figures == 1:
        images_dir = f"images_{class_type}/{now}"
        os.makedirs(f"{images_dir}", exist_ok=True)

        waveforms_dir = f"waveforms_{class_type}/{now}"
        os.makedirs(f"{waveforms_dir}", exist_ok=True)

    # Load audio files
    audio1, sr1 = load_audio_file(audio_file1)
    if print_figures == 1:
        print_waveform(
            audio1[0],
            sr1,
            f"Waveform ({class_type})\n{audio_file1}",
            f"{waveforms_dir}/{id}-waveform.png",
        )

    # Resample audio files
    audio1, sr1 = resample_audio(audio1, sr1, 16000)
    # normalize audio
    audio1 = normalize_waveform(audio1, max_loudness=max_loudness)
    if print_figures == 1:
        print_waveform(
            audio1[0],
            sr1,
            f"Resampled and Normalized Waveform ({class_type})\n{audio_file1}",
            f"{waveforms_dir}/{id}-waveform-resampled.png",
        )

    audio1 = vad_waveform(audio1[0], sr1)
    if print_figures == 1:
        print_waveform(
            audio1,
            sr1,
            f"VAD Waveform ({class_type})\n{audio_file1}",
            f"{waveforms_dir}/{id}-waveform-vad.png",
        )

    # Compute spectrograms
    if print_figures == 1:
        spectrogram1 = compute_mel_spectrogram(audio1, sr1, n_mels=n_mels)
        print_figure(
            f"{images_dir}/{id}-mel-spectrogram.png",
            f"Mel Spectrogram ({class_type})\n{audio_file1}",
            spectrogram1,
            sr1,
        )

    # Compute MFCCs
    audio_mfcc = audio_mfcc_transform(audio1, sr1, n_mfcc=n_mels)
    if print_figures == 1:
        print_figure(
            f"{images_dir}/{id}-mfcc.png",
            f"MFCC ({class_type})\n{audio_file1}",
            audio_mfcc.cpu().numpy(),
            sr1,
        )

    audio_mfcc_filename = f"{id}-{class_type}-mfcc.npy"
    np.save(f"{mfcc_dir}/{audio_mfcc_filename}", audio_mfcc)

    print(f"{class_type} MFCC computed and saved to {audio_mfcc_filename}")
    return audio_mfcc, audio_mfcc_filename, mfcc_dir


def main():

    parser = argparse.ArgumentParser(
        description="Compute distance between two audio files."
    )
    parser.add_argument("audio_file1", type=str, help="Path to the first audio file")
    parser.add_argument("id", type=int, help="id of the audio file", default=0)
    parser.add_argument(
        "now",
        type=str,
        help="timestamp",
        default=datetime.now().strftime("%Y-%m-%d--%H-%M-%S"),
    )
    parser.add_argument(
        "class_type", type=str, help="recorded or generated", default="recorded"
    )
    parser.add_argument(
        "print_figures", type=int, help="Print figures to file", default=0
    )
    parser.add_argument(
        "max_loudness", type=float, help="Max loudness of the audio files", default=None
    )
    parser.add_argument(
        "n_mels", type=int, help="Number of mfcc features to extract", default=13
    )
    args = parser.parse_args()
    print("args", args)
    audio_mfcc(
        args.audio_file1,
        args.id,
        args.now,
        args.class_type,
        args.print_figures,
        args.max_loudness,
        args.n_mels,
    )


if __name__ == "__main__":
    main()
