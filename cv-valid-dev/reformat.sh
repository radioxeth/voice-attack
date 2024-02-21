#!/bin/bash

# Directory containing the MP3 files
DIRECTORY="cv-valid-dev"

# Output directory for WAV files
OUTPUT_DIRECTORY="cv-valid-dev-wav"

# Check if the output directory exists, if not create it
if [ ! -d "$OUTPUT_DIRECTORY" ]; then
    mkdir -p "$OUTPUT_DIRECTORY"
fi

# Iterate through each MP3 file in the directory
for file in "$DIRECTORY"/*.mp3; do
    # Extract the filename without the extension
    filename=$(basename -- "$file" .mp3)
    
    # Define the output file path
    output_file="$OUTPUT_DIRECTORY/$filename.wav"
    
    # Convert MP3 to WAV using ffmpeg
    ffmpeg -i "$file" -acodec pcm_s16le -ar 44100 "$output_file"
    
    echo "Converted $file to $output_file"
done
