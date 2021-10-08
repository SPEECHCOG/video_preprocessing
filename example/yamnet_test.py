"""
This file is a simple script for running yamnet on a sample youtube video
"""

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.compat.v1.Session(config=config)

import numpy
import scipy
kh
# INPUT
# The model accepts a 1-D float32 Tensor or NumPy array containing a waveform of arbitrary length, represented as mono 16 kHz samples in the range [-1.0, +1.0]. 
# Internally, we frame the waveform into sliding windows of length 0.96 seconds and hop 0.48 seconds, and then run the core of the model on a batch of these frames.

# OUTPUT
# scores is a float32 Tensor of shape (N, 521) containing the per-frame predicted scores for each of the 521 classes in the AudioSet ontology that are supported by YAMNet.
# embeddings is a float32 Tensor of shape (N, 1024) containing per-frame embeddings, where the embedding vector is the average-pooled output that feeds into the final classifier layer.
# log_mel_spectrogram is a float32 Tensor representing the log mel spectrogram of the entire waveform. These are the audio features passed into the model and have shape (num_spectrogram_frames, 64) 
# where num_spectrogram_frames is the number of frames produced from the waveform by sliding a spectrogram analysis window of length 0.025 seconds with hop 0.01 seconds, and 64 represents the number of mel bins.
#%%############################################################################

"""
This is a simple script for reading audio from video
"""

import librosa
#import soundfile as sf

# wav_file_name = 'speech_whistling2.wav'
# sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
# sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)


def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform


# video sample
wav_file_name = "../../data/input/101_YSes0R7EksY.mp4"
#wav_file_name = "../data/input/101_3rtzSsuJ4Ng.mp4.webm"
wav_original, sample_rate = librosa.load(wav_file_name , mono=True)
#sf.write("../data/output/101_2Ihlw5FFrx4_sr.wav", wav_data, sample_rate)
wav_data = librosa.core.resample(wav_original, sample_rate, 16000) 

number_of_mel_bands = 64     
window_len_in_ms = 0.025
window_hop_in_ms = 0.01

win_len_sample = int (16000 * window_len_in_ms)
win_hop_sample = int (16000 * window_hop_in_ms)

mel_feature =  librosa.feature.melspectrogram(wav_data, sr=16000, n_fft=win_len_sample, hop_length=win_hop_sample, n_mels=number_of_mel_bands,power=2.0)
zeros_mel = mel_feature[mel_feature==0]          
if numpy.size(zeros_mel)!= 0:
    
    mel_flat = mel_feature.flatten('F')
    mel_temp =[value for counter, value in enumerate(mel_flat) if value!=0]

    if numpy.size(mel_temp)!=0:
        min_mel = numpy.min(numpy.abs(mel_temp))
    else:
        min_mel = 1e-12 
       
    mel_feature[mel_feature==0] = min_mel            
spectogram_librosa = numpy.transpose(10*numpy.log10(mel_feature))

# Show some basic information about the audio.
duration = len(wav_original)/sample_rate
duration = len(wav_data)/16000
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(wav_data)}')

#%%############################################################################

"""
loading yamnet model

"""


import tensorflow_hub as hub
import matplotlib.pyplot as plt
import csv


model = hub.load('https://tfhub.dev/google/yamnet/1')

window_len_in_ms = 0.96
window_hop_in_ms = 0.48

win_len_sample = int (16000 * window_len_in_ms)
win_hop_sample = int (16000 * window_hop_in_ms)

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

import pickle
with open("/worktmp/khorrami/project_5/video/data/youcook2/output/yamnet-based/yamnet_classes", 'wb') as handle:
    yamnet = pickle.dump(class_names,handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%############################################################################

"""
Executing the Model

"""

waveform = wav_data #/ tf.int16.max

# Run the model, check the output.
scores, embeddings, log_mel_spectrogram = model(waveform)

# asserting the output
scores.shape.assert_is_compatible_with([None, 521])
embeddings.shape.assert_is_compatible_with([None, 1024])
log_mel_spectrogram.shape.assert_is_compatible_with([None, 64])

# getting best scores
scores_np = scores.numpy()
spectrogram_np = log_mel_spectrogram.numpy()
embeddings_np = embeddings.numpy()
all_max_scores = scores_np.max(axis = 1)
all_max_class_indexes = scores_np.argmax(axis = 1)
all_max_class_names = [class_names[ind] for ind in all_max_class_indexes]

infered_class = class_names[scores_np.mean(axis=0).argmax()]
print(f'The main sound is: {infered_class}')

mean_scores = numpy.mean(scores, axis=0)
top_n = 10
top_class_indices = numpy.argsort(mean_scores)[::-1][:top_n]
top_classes = [class_names[ind] for ind in top_class_indices]
#%%############################################################################

"""
Plotting the results

"""

plt.figure(figsize=(10, 6))

# Plot the waveform.
plt.subplot(3, 1, 1)
plt.plot(waveform)
plt.xlim([0, len(waveform)])

# Plot the log-mel spectrogram (returned by the model).
plt.subplot(3, 1, 2)
plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')
#plt.imshow(spectogram_librosa.T, aspect='auto', interpolation='nearest', origin='lower')

# Plot and label the model output scores for the top-scoring classes.

plt.subplot(3, 1, 3)
plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')


# values from the model documentation
patch_padding = (0.025 / 2) / 0.01
plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
# Label the top_N classes.
yticks = range(0, top_n, 1)
plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])


#%%############################################################################

"""
converting the results from frames to seconds in the video

"""

win_hope = 0.48 # s/frame

position_frame = 446 # for example for the last frame
position_second = position_frame * win_hope # frame/ (frame/s)