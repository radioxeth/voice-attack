import argparse
import numpy as np
import torchaudio
import torch
from datetime import datetime
from util import (
    load_audio_file,
    compute_spectrogram,
    compute_euclidean_distances,
    compute_levenshtein_distance,
    center_and_pad_waveforms,
    print_euclidean_distance_figure,
    save_wav,
    print_waveform,
    print_figure,
)


def main():
    now = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    parser = argparse.ArgumentParser(
        description="Compute distance between two audio files."
    )
    parser.add_argument("audio_file1", type=str, help="Path to the first audio file")
    parser.add_argument("audio_file2", type=str, help="Path to the second audio file")
    # parser.add_argument("output_file", type=str, help="File name to save the output")

    args = parser.parse_args()
    print("args", args)
    # Load audio files
    audio1, sr1 = load_audio_file(args.audio_file1)
    audio2, sr2 = load_audio_file(args.audio_file2)

    print_waveform(audio1, sr1, "Waveform 1", f"waveforms/waveform1 {now}.png")
    print_waveform(audio2, sr2, "Waveform 2", f"waveforms/waveform2 {now}.png")
    audio1 = torch.tensor(audio1, dtype=torch.float32)
    audio2 = torch.tensor(audio2, dtype=torch.float32)

    # Ensure audio is in (channel, time) format if it's mono
    if audio1.dim() == 1:
        audio1 = audio1.unsqueeze(0)
    if audio2.dim() == 1:
        audio2 = audio2.unsqueeze(0)

    # detect voice activity
    audio1 = torchaudio.functional.vad(audio1, sr1)
    audio1 = torchaudio.functional.vad(audio2, sr2)

    if audio1.dim() == 2:
        audio1 = audio1.squeeze(0).numpy()
    if audio2.dim() == 2:
        audio2 = audio2.squeeze(0).numpy()

    print_waveform(audio1, sr1, "VAD Waveform 1", f"waveforms/vad_waveform1 {now}.png")
    print_waveform(audio2, sr2, "VAD Waveform 2", f"waveforms/vad_waveform2 {now}.png")
    audio1, audio2 = center_and_pad_waveforms(audio1, sr1, audio2, sr2)

    print_waveform(
        audio1,
        sr1,
        "Padded VAD Waveform 1",
        f"waveforms/padded_vad_waveform1 {now}.png",
    )
    print_waveform(
        audio2,
        sr2,
        "Padded VAD Waveform 2",
        f"waveforms/padded_vad_waveform2 {now}.png",
    )
    output_file_vad1 = f"generated_audio/vad1 {now}.wav"
    output_file_vad2 = f"generated_audio/vad2 {now}.wav"
    save_wav(audio1, sr1, output_file_vad1)
    save_wav(audio1, sr1, output_file_vad2)

    # Compute spectrograms
    spectrogram1 = compute_spectrogram(audio1, sr1)
    spectrogram2 = compute_spectrogram(audio2, sr2)
    print_figure(
        f"generated_images/spectrogram1 {now}.png", "spectrogram1", spectrogram1, sr1
    )
    print_figure(
        f"generated_images/spectrogram2 {now}.png", "spectrogram2", spectrogram2, sr2
    )

    print_euclidean_distance_figure(
        f"generated_images/euc_distance {now}.png",
        "Euclidean Distance of Mel Spectrograms",
        spectrogram1,
        spectrogram2,
        sr1,
        filter_level=0,
    )

    # Compute distance
    distances = compute_euclidean_distances(audio1, audio2)

    output_file = f"generated_audio/euc_distance {now}.wav"
    save_wav(distances, sr1, output_file)

    print(f"Distance computed and saved to {output_file}")


if __name__ == "__main__":
    main()
