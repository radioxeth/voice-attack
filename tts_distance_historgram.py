import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv("audio_generated/tts2024-03-13--08-38-57/results.csv")

# find the mean tts_distance
mean = df["tts_distance"].mean()
print(mean)

# find the standard deviation
std = df["tts_distance"].std()
print(std)


# Step 2: Plot the Histogram using matplotlib
plt.figure(figsize=(10, 6))
plt.hist(df["tts_distance"], bins=50)
plt.title("Histogram of TTS Levenshtein Distance")
plt.xlabel("Levenshtein Distance")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("tts_distance_histogram_og.png")
plt.close()

# remove outliers of 2 std
# outliers = df[df["tts_distance"] >= mean + 2 * std]
df = df[df["tts_distance"] < mean + 2 * std]

# Step 2: Plot the Histogram using matplotlib
plt.figure(figsize=(10, 6))
plt.hist(df["tts_distance"], bins=50)
plt.title("Histogram of TTS Levenshtein Distance (outliers removed)")
plt.xlabel("Levenshtein Distance")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("tts_distance_histogram.png")
plt.close()
