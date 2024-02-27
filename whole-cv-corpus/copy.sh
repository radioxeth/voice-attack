#!/bin/bash

INPUT=./validated_head.tsv
OUTPUT=./validated_output.tsv
DIRECTORY_PATH=./clips
COLUMN=2 # Change this to the column number containing the filenames

echo "$DIRECTORY_PATH"

# Check if OUTPUT file exists and remove it to start fresh
if [ -f "$OUTPUT" ]; then
    rm "$OUTPUT"
fi

while IFS=$'\t' read -r -a line; do
    # check if it is the first line
    if [ "${line[0]}" = "client_id" ]; then
        IFS=$'\t'; echo "${line[*]}" >> "$OUTPUT"
        continue
    fi
    
    FILENAME="${line[$((COLUMN-1))]}" # Adjust column index for 0-based indexing in arrays
    if [ -f "$DIRECTORY_PATH/$FILENAME" ]; then
        # append line to output file as a comma separated string
        IFS=$'\t'; echo "${line[*]}" >> "$OUTPUT"
    fi
done < "$INPUT"


