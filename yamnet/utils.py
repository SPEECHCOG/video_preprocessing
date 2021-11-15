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


def resample_audio (wav_original, sample_rate):
    wav_data = librosa.core.resample(wav_original, sample_rate, 16000) 
    return wav_data


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

"""
calculating log-mel features

"""

def calculate_logmels (y , number_of_mel_bands , window_len_in_ms , window_hop_in_ms , sr_target):
    
    win_len_sample = int (sr_target * window_len_in_ms)
    win_hop_sample = int (sr_target * window_hop_in_ms)
      
    mel_feature = librosa.feature.melspectrogram(y=y, sr=sr_target, n_fft=win_len_sample, hop_length=win_hop_sample, n_mels=number_of_mel_bands,power=2.0)
    #print('.......... mel features are found ..............')
    zeros_mel = mel_feature[mel_feature==0]          
    if numpy.size(zeros_mel)!= 0:
        
        mel_flat = mel_feature.flatten('F')
        mel_temp =[value for counter, value in enumerate(mel_flat) if value!=0]
    
        if numpy.size(mel_temp)!=0:
            min_mel = numpy.min(numpy.abs(mel_temp))
        else:
            min_mel = 1e-12 
           
        mel_feature[mel_feature==0] = min_mel           
    logmel_feature = numpy.transpose(10*numpy.log10(mel_feature)) 
    #print('..........log mel features are found ..............')  
    #print (numpy.shape(logmel_feature))    
    return logmel_feature
