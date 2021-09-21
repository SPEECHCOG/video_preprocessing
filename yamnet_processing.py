"""
This file is a script for running yamnet for detecting onsets of speech segments 
"""

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.compat.v1.Session(config=config)

import numpy
import scipy


#%%############################################################################

"""
This is a simple script for reading audio from video
"""

import librosa
#import soundfile as sf


# video sample
wav_file_name = "../data/input/101_2Ihlw5FFrx4.mp4"
wav_original, sample_rate = librosa.load(wav_file_name , mono=True)
#sf.write("../data/output/101_2Ihlw5FFrx4_sr.wav", wav_data, sample_rate)
wav_data = librosa.core.resample(wav_original, sample_rate, 16000) 

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
class_names_accepted = ['Speech', 'Child speech, kid speaking' , 'Conversation', 'Narration, monologue' ]
class_index_accepted = [0,1,2,3]
#%%############################################################################

"""
Executing the Model

"""

waveform = wav_data 

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

# detecting speech segments

win_hope = 0.48 # s/frame
position_frame = 446 # for example for the last frame
position_second = position_frame * win_hope # frame/ (frame/s)


speech_segments_zero = numpy.zeros(len(all_max_class_indexes))
speech_segments =  [item == 0 or item == 1 or item == 2 or item == 3 for item in all_max_class_indexes]
speech_segments = numpy.multiply(speech_segments , 1)
plt.plot(speech_segments)

clip_length_seconds = 10
clip_length_frames = int (round(clip_length_seconds / win_hope ))  # 21 frames --> almost equal to ~10 seconds of audio
accepted_rate = 0.8
accepted_frame_numebrs = int(round(21 * 0.8)) # 17

onset_candidate = 445 
clip_candidate = speech_segments [onset_candidate:onset_candidate + clip_length_frames]
clip_candidate_pass = numpy.sum(clip_candidate) >= accepted_frame_numebrs



# seed the pseudorandom number generator
# from random import seed
# from random import randint
# seed random number generator
# seed(1)
# print(randint(0,20), randint(0,20), randint(0,20))

from random import choice
sequence = [onset for onset in range(clip_length_frames , len(speech_segments) - clip_length_frames)] # skip first 10 seconds of the audio which is 21 frames
selection = choice(sequence)

max_trials = int( len(sequence) / 2)
max_len_accepted_onsets = int( len(sequence) * 0.1)

trial_number = 0
accepted_onsets = []
while( trial_number < max_trials):
    trial_number += 1
    onset_candidate = choice(sequence)
    clip_candidate = speech_segments [onset_candidate:onset_candidate + clip_length_frames]
    if numpy.sum(clip_candidate) >= accepted_frame_numebrs:        
        accepted_onsets.append(onset_candidate)
    
    if len(accepted_onsets) >= max_len_accepted_onsets:
        break

