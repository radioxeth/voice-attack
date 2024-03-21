# voice-attack
Using k-means and naive bayes models to classify human and machine generated audio samples.

- `utils.py` contains many reusable and helpful functions
- `tts.py` to generate text to speech samples from cv-valid-dev-wav
- `tts_distance_histogram.py` to generate the histogram of the levenshtein distances
- `audio_mfcc.py` to extract MFCC features from audio files
  - `process_audio_mfcc.py` to iterate and extract MFCCs from many audio files
- `k_means.py` to perform k-means on generated and recorded audio for 1 MFCC
  - `process_k_means.py` to run k-means on generated and recorded audio for many MFCCs
- `naive_bayes.py` to perform naive bayes on generated and recorded audio for 1 MFCC
  - `process_naive_bayes.py` to run naive bayes on generated and recorded audio for many MFCCs
- `analyze_kmeans.py` and `analyze_naive_bayes.py` analyzes the results of the respective models and produce performance metrics
- `roc_curve_compare.py` to generate roc curve and comparrison scatter chart.