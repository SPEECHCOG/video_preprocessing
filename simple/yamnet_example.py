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



# video sample
wav_file_name = "/worktmp2/hxkhkh/current/video/data/example/input/train/108_soByLvI6SSY.mp4"
wav_original, sample_rate = librosa.load(wav_file_name , mono=True)

wav_data = librosa.core.resample(wav_original, sample_rate, 16000) 

# Show some basic information about the audio.
duration = len(wav_original)/sample_rate
duration = len(wav_data)/16000
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(wav_data)}')


def calculate_logmels (y , number_of_mel_bands , window_len_in_ms , window_hop_in_ms , sr_target):
    
    win_len_sample = int (sr_target * window_len_in_ms)
    win_hop_sample = int (sr_target * window_hop_in_ms)
      
    mel_feature = librosa.feature.melspectrogram(y=y, sr=sr_target, n_fft=win_len_sample, hop_length=win_hop_sample, n_mels=number_of_mel_bands,power=2.0)
    
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
    return logmel_feature

my_logmels = calculate_logmels (wav_data , 40 , 0.025 , 0.01 , 16000)
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

#%%############################################################################

"""
Detecting speech segments

""" 
win_hope_logmel = 0.01

win_hope_yamnet = 0.48 # s/frame
position_frame = 446 # for example for the last frame
position_second = position_frame * win_hope_yamnet # frame/ (frame/s)


speech_segments =  [item == 0 or item == 1 or item == 2 or item == 3 for item in all_max_class_indexes]
speech_segments = numpy.multiply(speech_segments , 1)
plt.plot(speech_segments)

clip_length_seconds = 10
clip_length_yamnet = int (round(clip_length_seconds / win_hope_yamnet ))  # 21 frames --> almost equal to ~10 seconds of audio
clip_length_logmel = int (round(clip_length_seconds / win_hope_logmel)) # 1000 frames


from random import shuffle

accepted_rate = 0.8
accepted_plus = int(round(clip_length_yamnet * 0.8)) # e.g. for clip of 10 seconds: 21* 0.8 = 17

number_of_clips = int( duration / clip_length_seconds) 

start_frame = 21 # skip first 10 seconds of the audio which is 21 yamnet frames
end_frame = len(speech_segments) - clip_length_yamnet

skip_seconds = 3
skip_frames_yamnet = int(round(skip_seconds/ win_hope_yamnet))
initial_sequence = [ onset for onset in range(start_frame , end_frame , 6) ]  # sample every 3 * one second (which is ~2 yamnet frames)

trial_index = len(initial_sequence) -1 
accepted_onsets_yamnet = []
upated_sequence = initial_sequence [:]
shuffle(upated_sequence)

# scan from end to start not to loose any member due to updated list
while( trial_index >=  0):
    
    onset_candidate = upated_sequence [trial_index] # choice(upated_sequence)
    trial_index -= 1    
    upated_sequence.remove(onset_candidate) # remove choice from upated_sequence  
    clip_candidate = speech_segments [onset_candidate: onset_candidate + clip_length_yamnet]
    if numpy.sum(clip_candidate) >= accepted_plus:        
        accepted_onsets_yamnet.append(onset_candidate)  
    if len(accepted_onsets_yamnet) >= number_of_clips:
        break

print(accepted_onsets_yamnet)
import math
accepted_onsets_second = [math.floor(item * win_hope_yamnet) for item in accepted_onsets_yamnet]
accepted_onset_logmel = [math.floor(item / win_hope_logmel) for item in accepted_onsets_second]
#%%############################################################################

"""
Collecting log-mel features for selected onsets

"""


accepted_logmels = [spectrogram_np[onset:onset + clip_length_logmel] for onset in accepted_onset_logmel]
accepted_mylogmels = [my_logmels[onset:onset + clip_length_logmel] for onset in accepted_onset_logmel] 

accepted_embeddings = [embeddings_np[onset:onset + clip_length_yamnet] for onset in accepted_onsets_yamnet]


#%%############################################################################

"""
Saving results for candidate onsets in one dictionary

"""

import pickle

dict_out = {}
dict_out[wav_file_name] = accepted_onsets_second


outputfile = "/worktmp2/hxkhkh/current/video/data/example/output/onsets_testiiing"

with open(outputfile, 'wb') as handle:
    pickle.dump(dict_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open("/worktmp2/hxkhkh/current/video/data/example/output/", 'rb') as handle:
    b = pickle.load(handle)
    
#%%############################################################################

"""
Saving audio features each in a seprate dictionary

"""


import os

counter = 1
output_path = "/worktmp2/hxkhkh/current/video/data/example/output/" + str(counter) 
os.mkdir(output_path)
output_name =  output_path + '/af'
dict_out = {}
dict_out['video_name'] = wav_file_name
dict_out['onsets_second'] = accepted_onsets_second
dict_out['logmel40'] = accepted_mylogmels
dict_out['logmel64'] = accepted_logmels
dict_out['embeddings'] = accepted_embeddings

with open(output_name, 'wb') as handle:
    pickle.dump(dict_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(output_name, 'rb') as handle:
    c = pickle.load(handle)
    
#%%############################################################################

"""
Saving audio features each in a seprate dictionary

"""

import soundfile as sf
duration = len(wav_data)/16000
for second in accepted_onsets_second:

    wav_clip = wav_data[int(second *16000 ): int((second + 10)*16000 )]
    bool_check = sf.write("/worktmp2/hxkhkh/current/video/data/example/output/" + str(int(second)) + ".wav", wav_clip, 16000)