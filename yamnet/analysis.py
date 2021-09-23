import os
import librosa
import numpy
from random import shuffle
import math
import pickle

import utils

import tensorflow_hub as hub

class Analysis:
    
    def __init__(self,audio_model,dataset, datadir, outputdir, split , yamnet_settings):
        
        self.audio_model = audio_model
        self.dataset = dataset
        self.datadir = datadir
        self.outputdir = outputdir
        self.split = split
        
        self.yamnet_settings = yamnet_settings
        
        self.clip_length_seconds = self.yamnet_settings ["clip_length_seconds"]
        self.win_hope_yamnet = self.yamnet_settings ["win_hope_yamnet"] 
        self.win_hope_logmel = self.yamnet_settings ["win_hope_logmel"]
        
        self.clip_length_yamnet = int (round(self.clip_length_seconds / self.win_hope_yamnet ))
        self.clip_length_logmel = int (round(self.clip_length_seconds / self.win_hope_logmel))
        
        self.counter = 0
        self.video_name = ''
        self.video_duration = 0

        self.dict_onsets = {}
        
    def create_video_list (self ):
        # use npath and take name of all videos
        # return video name list
        video_dir = os.path.join(self.datadir, self.split)
        video_list = os.listdir(video_dir)
        return video_list
    
    def load_video (self):
        
        video_name = self.video_name
        video_path = os.path.join(self.datadir,self.split, video_name) 
        target_sr = self.yamnet_settings ["target_sample_rate"]
        wav_data, sample_rate = librosa.load(video_path , sr=target_sr , mono=True)        
        duration = len(wav_data)/target_sr
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

    
    def execute_yamnet (self, wavedata ):
          
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        scores, embeddings, log_mel_yamnet = model(wavedata)       
        scores_np = scores.numpy()
        logmel_yamnet_np = log_mel_yamnet.numpy()
        embeddings_np = embeddings.numpy()

        return scores_np, embeddings_np, logmel_yamnet_np

    
    def detect_speech_frames (self, scores_yamnet):
        
        all_max_class_indexes = scores_yamnet.argmax(axis = 1)        
        #class_index_accepted = self.yamnet_settings [ "class_index_accepted" ]
        speech_segments =  [item == 0 or item == 1 or item == 2 or item == 3 for item in all_max_class_indexes]
        speech_segments = numpy.multiply(speech_segments , 1)
        return speech_segments

    def initialize_global_params (self):
        pass
        
    def produce_onset_candidates (self, speech_segments):
               
        clip_length_seconds = self.clip_length_seconds
        
        accepted_rate = self.yamnet_settings ["acceptance_snr"]
        skip_seconds = self.yamnet_settings ["skip_seconds"]
        # e.g., 21 frames --> almost equal to ~10 seconds of audio
        # and, for clip of 10 seconds: 21* 0.8 = 17
        # also, skip first 10 seconds of the audio which is 21 yamnet frames
        # sample every 3 seconds (which is ~ 3* 2 yamnet frames)
              
        clip_length_yamnet = self.clip_length_yamnet        
        accepted_plus = int(round(clip_length_yamnet * accepted_rate)) 
        number_of_clips = int( self.video_duration / clip_length_seconds) 
        
        start_frame = clip_length_yamnet 
        end_frame = len(speech_segments) - clip_length_yamnet
        skip_frames_yamnet = int(round(skip_seconds/ self.win_hope_yamnet))
        initial_sequence = [ onset for onset in range(start_frame , end_frame , skip_frames_yamnet) ]  
        
        trial_index = len(initial_sequence) -1 
        onsets_yamnet = []
        upated_sequence = initial_sequence [:]
        shuffle(upated_sequence)
        
        # scan from end to start not to loose any member due to updated list
        while( trial_index >=  0):
            
            onset_candidate = upated_sequence [trial_index] # choice(upated_sequence)
            trial_index -= 1    
            upated_sequence.remove(onset_candidate) # remove choice from upated_sequence  
            clip_candidate = speech_segments [onset_candidate: onset_candidate + clip_length_yamnet]
            if numpy.sum(clip_candidate) >= accepted_plus:        
                onsets_yamnet.append(onset_candidate)  
            if len(onsets_yamnet) >= number_of_clips:
                break
        
        print('###############################################################')
        
        print(self.counter)
        print(self.video_name)
        print(self.video_duration)
        print(onsets_yamnet)
        
        print('###############################################################')
         
        onsets_second = [math.floor(item * self.win_hope_yamnet) for item in onsets_yamnet]
        onsets_logmel = [math.floor(item / self.win_hope_logmel) for item in onsets_second]
        return onsets_yamnet , onsets_second , onsets_logmel 
    
    def convert_frame_to_seconds (self):
        # either here or in utils
        pass
    
    def convert_second_to_frame (self):
        # either here or in utils
        pass
    
    def update_onset_list (self, accepted_onsets_second):
        
        self.dict_onsets[self.video_name] = {'onsets': accepted_onsets_second , 'folder_name':self.counter}
        
    
    def save_per_video (self, logmel_yamnet, embeddings_yamnet, logmels , onsets_yamnet , onsets_second , onsets_logmel):
        # save features and onsets for each video in video path  
        clip_length_yamnet = self.clip_length_yamnet
        clip_length_logmel = self.clip_length_logmel
        
        accepted_yamnetlogmels = [logmel_yamnet[onset:onset + clip_length_logmel] for onset in onsets_logmel]
        accepted_logmels = [logmels[onset:onset + clip_length_logmel] for onset in onsets_logmel] 
        accepted_embeddings = [embeddings_yamnet[onset:onset + clip_length_yamnet] for onset in onsets_yamnet]
        
        
        output_path = self.outputdir + str(self.counter)
        os.mkdir(output_path)
        output_name = output_path  + '/af'
        dict_out = {}
        dict_out['video_name'] = self.video_name
        dict_out['video_duration'] = self.video_duration
        dict_out['onsets_second'] = onsets_second
        dict_out['logmel40'] = accepted_logmels
        dict_out['logmel64'] = accepted_yamnetlogmels
        dict_out['embeddings'] = accepted_embeddings
        
        
        with open(output_name, 'wb') as handle:
            pickle.dump(dict_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
          
        
    
    def __call__ (self):
        # call above functions one by one
        video_list = self.create_video_list()
        
        for video_name in video_list:         
            self.video_name = video_name
            # do all analysis
            wav_data = self.load_video ()
            logmels = self.extract_logmel_features (wav_data)
            scores_yamnet, embeddings_yamnet, logmel_yamnet = self.execute_yamnet(wav_data)
            speech_segments = self.detect_speech_frames(scores_yamnet)
            onsets_yamnet , onsets_second , onsets_logmel = self.produce_onset_candidates (speech_segments)
            self.update_onset_list (onsets_second)
            self.save_per_video ( logmel_yamnet, embeddings_yamnet, logmels , onsets_yamnet , onsets_second , onsets_logmel)
            
            self.counter += 1
            
        output_name =  self.outputdir + self.split + '_onsets'  
        with open(output_name , 'wb') as handle:
            pickle.dump(self.dict_onsets, handle, protocol=pickle.HIGHEST_PROTOCOL)