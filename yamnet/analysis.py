import os
import librosa
import numpy
from random import shuffle
import math
import pickle

import utils

import tensorflow_hub as hub

class Analysis:
    
    def __init__(self,audio_model,dataset, datadir,outputdir,split , yamnet_settings):
        
        self.audio_model = audio_model
        self.dataset = dataset
        self.datadir = datadir
        self.outputdir = outputdir
        self.split = split
        
        self.yamnet_settings = yamnet_settings
        
        self.onset_list = []
        self.counter = 1
        self.video_duration = 0
        
    def create_video_list (self ):
        # use npath and take name of all videos
        # return video name list
        video_list = os.listdir(self.datadir)
        return video_list
    
    def load_video (self , video_name):
        
        video_path = os.path.join(self.datadir, video_name)      
        wav_data, sample_rate = librosa.load(video_path , sr=16000 , mono=True)        
        duration = len(wav_data)/16000
        self.video_duration = duration
        return wav_data
    
    def resample_audio (self):
        #put this either here or in utilsimport math
        pass
    
    def extract_logmel_features (self, wav_data):
        window_len_in_ms = self.yamnet_settings ['win_length_logmel']
        window_hop_in_ms = self.yamnet_settings ['win_hope_logmel']
        number_of_mel_bands = self.yamnet_settings ['logmel_bands']
        sr_target = self.yamnet_settings ['target_sample_rate' ]
        logmels = utils.calculate_logmels (wav_data , number_of_mel_bands , window_len_in_ms , window_hop_in_ms , sr_target)        
        return logmels
    
    def execute_yamnet (self, video_name):
        # load model and find speech segments
        # get model output + spectogram + embeddings
        wavedata = self.load_video (video_name)
               
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        scores, embeddings, log_mel_yamnet = model(wavedata)
        
        scores_np = scores.numpy()
        logmel_yamnet_np = log_mel_yamnet.numpy()
        embeddings_np = embeddings.numpy()

        return scores_np, embeddings_np, logmel_yamnet_np
    
    def detect_speech_frames (self, video_name):

        scores_np, embeddings_np, logmel_yamnet_np = self.execute_yamnet (video_name)       
        all_max_class_indexes = scores_np.argmax(axis = 1)        
        #class_index_accepted = self.yamnet_settings [ "class_index_accepted" ]
        speech_segments =  [item == 0 or item == 1 or item == 2 or item == 3 for item in all_max_class_indexes]
        speech_segments = numpy.multiply(speech_segments , 1)
        return speech_segments, 
    
    def produce_onset_candidates (self, video_name):
        # use 0.8 to produce onset candidates for clips of len 10 seconds
        
        speech_segments = self.detect_speech_frames (video_name)
        clip_length_seconds = 10
        win_hope_yamnet = 0.48
        win_hope_logmel = 0.01
        
        clip_length_yamnet = int (round(clip_length_seconds / win_hope_yamnet ))  # 21 frames --> almost equal to ~10 seconds of audio
        #clip_length_logmel = int (round(clip_length_seconds / win_hope_logmel)) # 1000 frames
        
        
        accepted_rate = 0.8
        accepted_plus = int(round(clip_length_yamnet * accepted_rate)) # 17
               
        initial_sequence = [onset for onset in range(clip_length_yamnet , len(speech_segments) - clip_length_yamnet)] # skip first 10 seconds of the audio which is 21 frames
            
        max_trials = int( len(initial_sequence) / 2)
        max_number_of_clips = int( self.video_duration / clip_length_seconds)
        
        trial_number = 0
        accepted_onsets_yamnet = []
        upated_sequence = initial_sequence [:]
        shuffle(upated_sequence)
        
        while( trial_number < max_trials):
            
            onset_candidate = upated_sequence [trial_number] # choice(upated_sequence)
            trial_number += 1    
            upated_sequence.remove(onset_candidate) # remove choice from upated_sequence
            
            clip_candidate = speech_segments [onset_candidate:onset_candidate + clip_length_yamnet]
            if numpy.sum(clip_candidate) >= accepted_plus:        
                accepted_onsets_yamnet.append(onset_candidate)
            
            if len(accepted_onsets_yamnet) >= max_number_of_clips:
                break
        
        print(accepted_onsets_yamnet)
        
        accepted_onsets_second = [math.floor(item * win_hope_yamnet) for item in accepted_onsets_yamnet]
        accepted_onset_logmel = [math.floor(item / win_hope_logmel) for item in accepted_onsets_second]
        return accepted_onsets_yamnet , accepted_onsets_second , accepted_onset_logmel
    
    def convert_frame_to_seconds (self):
        # either here or in utils
        pass
    
    def convert_second_to_frame (self):
        # either here or in utils
        pass
    
    def update_onset_list (self, video_name):
        # add results after each video
        accepted_onsets_yamnet , accepted_onsets_second , accepted_onset_logmel = self.produce_onset_candidates (self, video_name)
        self.onset_list.append(accepted_onsets_second)
    
    def save_per_video (self, video_name , spectrogram_np, embeddings_np, logmels):
        # save features and onsets for each video in video path
        accepted_onsets_yamnet , accepted_onsets_second , accepted_onset_logmel = self.produce_onset_candidates (self, video_name)
        
        clip_length_seconds = 10
        win_hope_yamnet = 0.48
        win_hope_logmel = 0.01
        
        clip_length_yamnet = int (round(clip_length_seconds / win_hope_yamnet ))  # 21 frames --> almost equal to ~10 seconds of audio
        clip_length_logmel = int (round(clip_length_seconds / win_hope_logmel)) # 1000 frames
        
        
        accepted_yamnetlogmels = [spectrogram_np[onset:onset + clip_length_logmel] for onset in accepted_onset_logmel]
        accepted_logmels = [logmels[onset:onset + clip_length_logmel] for onset in accepted_onset_logmel] 
        accepted_embeddings = [embeddings_np[onset:onset + clip_length_yamnet] for onset in accepted_onsets_yamnet]
        
        counter = self.counter
        output_path ="../data/output/" + "train/" + str(counter) 
        os.mkdir(output_path)
        output_name =  output_path + '/af'
        dict_out = {}
        dict_out['video_name'] = video_name
        dict_out['onsets_second'] = accepted_onsets_second
        dict_out['logmel40'] = accepted_logmels
        dict_out['logmel64'] = accepted_yamnetlogmels
        dict_out['embeddings'] = accepted_embeddings
        
        with open(output_name, 'wb') as handle:
            pickle.dump(dict_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.counter = counter + 1   
        
    
    def __call__ (self):
        # call above functions one by one
        video_list = self.create_video_list()
        for video_name in video_list:
            # do all analysis
            a = 1
        