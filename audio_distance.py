import os
import argparse
import numpy as np
import torchaudio
import librosa
import torch
from datetime import datetime
from util import (
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
)


def audio_distance(
    audio_file1, audio_file2, id=0, now=datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
):

    generated_audio_dir = f"generated_audio/{now}"
    os.makedirs(f"{generated_audio_dir}", exist_ok=True)

    generated_images_dir = f"generated_images/{now}"
    os.makedirs(f"{generated_images_dir}", exist_ok=True)

    waveforms_dir = f"waveforms/{now}"
    os.makedirs(f"{waveforms_dir}", exist_ok=True)

    # Load audio files
    audio1, sr1 = load_audio_file(audio_file1)
    audio2, sr2 = load_audio_file(audio_file2)

    # Resample audio files
    audio1, sr1 = resample_audio(audio1, sr1, 16000)
    audio2, sr2 = resample_audio(audio2, sr2, 16000)

    # print_waveform(audio1, sr1, "Waveform 1", f"waveforms/waveform1 {now}.png")
    # print_waveform(audio2, sr2, "Waveform 2", f"waveforms/waveform2 {now}.png")
    audio1 = vad_waveform(audio1[0], sr1)
    audio2 = vad_waveform(audio2[0], sr2)

    print_waveform(
        audio1, sr1, "VAD Waveform 1", f"{waveforms_dir}/{id}-vad_waveform1-{now}.png"
    )
    print_waveform(
        audio2, sr2, "VAD Waveform 2", f"{waveforms_dir}/{id}-vad_waveform2-{now}.png"
    )
    audio1, audio2 = center_and_pad_waveforms(audio1, sr1, audio2, sr2)

    # print_waveform(
    #     audio1,
    #     sr1,
    #     f"Padded VAD Waveform {audio_file1}",
    #     f"{waveforms_dir}/{id}-padded_vad_waveform1-{now}.png",
    # )
    # print_waveform(
    #     audio2,
    #     sr2,
    #     f"Padded VAD Waveform {audio_file2}",
    #     f"{waveforms_dir}/{id}-padded_vad_waveform2-{now}.png",
    # )
    # mfcc1 = audio_mfcc_transform(audio1, sr1, n_mels=80)
    # mfcc2 = audio_mfcc_transform(audio2, sr2, n_mels=80)

    # print(f"MFCC 1 {audio_file1}", mfcc1)
    # print(f"MFCC 2 {audio_file2}", mfcc2)

    # Compute spectrograms
    spectrogram1 = compute_mel_spectrogram(audio1, sr1, n_mels=40)
    spectrogram2 = compute_mel_spectrogram(audio2, sr2, n_mels=40)
    print_figure(
        f"{generated_images_dir}/{id}-spectrogram1-{now}.png",
        f"{audio_file1} mel spectrogram",
        spectrogram1,
        sr1,
    )

    print_figure(
        f"{generated_images_dir}/{id}-spectrogram2-{now}.png",
        f"{audio_file2} mel spectrogram",
        spectrogram2,
        sr2,
    )

    print_euclidean_distance_figure(
        f"{generated_images_dir}/{id}-euc_distance-{now}.png",
        f"Euclidean Distance of Mel Spectrograms - {audio_file1} vs {audio_file2}",
        spectrogram1,
        spectrogram2,
        sr1,
        filter_level=10,
    )

    # Compute distance
    audio_distance = compute_euclidean_distances(audio1, audio2)

    # print_waveform(
    #     audio_distance,
    #     sr1,
    #     f"Euclidean Distance {id}",
    #     f"{waveforms_dir}/{id}-euc_distance-{now}.png",
    # )
    output_file = f"{generated_audio_dir}/{id}-euc_distance-{now}.wav"
    save_wav(audio_distance, sr1, output_file)

    print(f"Distance computed and saved to {output_file}")
    return output_file


def main():

    parser = argparse.ArgumentParser(
        description="Compute distance between two audio files."
    )
    parser.add_argument("audio_file1", type=str, help="Path to the first audio file")
    parser.add_argument("audio_file2", type=str, help="Path to the second audio file")
    # parser.add_argument("output_file", type=str, help="File name to save the output")

    args = parser.parse_args()
    print("args", args)
    audio_distance(args.audio_file1, args.audio_file2)


if __name__ == "__main__":
    main()
