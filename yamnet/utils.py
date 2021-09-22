"""
This file includes different tools including

1. A function for loading yamnet model
2. A function for changing sample rate of an audio signal 
"""

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.compat.v1.Session(config=config)

import numpy
import scipy
import tensorflow_hub as hub
import csv
import librosa

"""
This is a simple script for reading audio from video
"""


#import soundfile as sf

# wav_file_name = 'speech_whistling2.wav'
# sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
# sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

"""
changing sample rate

"""

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform



"""
loading yamnet model

"""

def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

def load_yamnet_model():

    model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    #window_len_in_ms = 0.96
    #window_hop_in_ms = 0.48  
    #win_len_sample = int (16000 * window_len_in_ms)
    #win_hop_sample = int (16000 * window_hop_in_ms)
 
    class_map_path = model.class_map_path().numpy()
    class_names = class_names_from_csv(class_map_path)
    
    return model, class_map_path, class_names



