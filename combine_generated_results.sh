#!/bin/bash

# Combine the generated results from the different models
# gather the results.csv file from each directory in generated_audio that begins with tts
# and combine them into a single file

# The directory containing the generated results
DIRECTORY_PATH=./generated_audio

# The output file
OUTPUT=./generated_results.csv

# Check if OUTPUT file exists and remove it to start fresh
if [ -f "$OUTPUT" ]; then
    rm "$OUTPUT"
fi

echo "id,pair_id,text,transcript,filename,tts_distance" >> "$OUTPUT"
# Recursively loop through each directory in DIRECTORY_PATH
for dir in $DIRECTORY_PATH/tts*; do
    # Check if the directory exists
    if [ -d "$dir" ]; then
        # Check if the results.csv file exists in the directory
        if [ -f "$dir/results.csv" ]; then
            # Append the results.csv file to the output file
            # do not include the header
            tail -n +2 "$dir/results.csv" >> "$OUTPUT"
        fi
    fi
done

