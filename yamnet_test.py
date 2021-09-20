"""
This is a simple script for running yamnet on videos
"""

import numpy
import scipy

#%%
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.compat.v1.Session(config=config)


import tensorflow_hub as hub
import csv

# Load the model.
model = hub.load('https://tfhub.dev/google/yamnet/1')

#%%

import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)


#%%

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform


#%%
import librosa
import soundfile as sf
# wav_file_name = 'speech_whistling2.wav'
# sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
# sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)


# video sample
wav_file_name = "../data/input/101_2Ihlw5FFrx4.mp4"
#wav_file_name = "../data/input/101_3rtzSsuJ4Ng.mp4.webm"
wav_data, sample_rate = librosa.load(wav_file_name , sr = 16000)
#sf.write("../data/output/101_2Ihlw5FFrx4_sr.wav", wav_data, sample_rate)


# Show some basic information about the audio.
duration = len(wav_data)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(wav_data)}')

# Listening to the wav file.
Audio(wav_data, rate=sample_rate)

#%%

# Executing the Model

waveform = wav_data #/ tf.int16.max

# Run the model, check the output.
scores, embeddings, spectrogram = model(waveform)

scores_np = scores.numpy()
spectrogram_np = spectrogram.numpy()
infered_class = class_names[scores_np.mean(axis=0).argmax()]
print(f'The main sound is: {infered_class}')


#%%

plt.figure(figsize=(10, 6))

# Plot the waveform.
plt.subplot(3, 1, 1)
plt.plot(waveform)
plt.xlim([0, len(waveform)])

# Plot the log-mel spectrogram (returned by the model).
plt.subplot(3, 1, 2)
plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')

# Plot and label the model output scores for the top-scoring classes.
mean_scores = numpy.mean(scores, axis=0)
top_n = 10
top_class_indices = numpy.argsort(mean_scores)[::-1][:top_n]
plt.subplot(3, 1, 3)
plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')

# patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
# values from the model documentation
patch_padding = (0.025 / 2) / 0.01
plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
# Label the top_N classes.
yticks = range(0, top_n, 1)
plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
# _ = plt.ylim(-0.5 + np.array([top_n, 0]))
