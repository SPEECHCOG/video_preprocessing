import os
import librosa
import numpy
from random import shuffle
import soundfile as sf
import math
import pickle
import copy
import utils
import tensorflow_hub as hub

class Analysis:
    
    def __init__(self,audio_model,dataset, datadir, outputdir, split ,save_wavs, yamnet_settings, rsd, exp_name):
        
        self.audio_model = audio_model
        self.dataset = dataset
        self.datadir = datadir
        self.save_wavs = save_wavs
        self.outputdir = outputdir
        self.exp_name = exp_name
        self.split = split
        self.run_speech_detection = rsd
        
        self.yamnet_settings = yamnet_settings
        self.target_sr = self.yamnet_settings ["target_sample_rate"] 
        self.clip_length_seconds = self.yamnet_settings ["clip_length_seconds"]
        self.win_hope_yamnet = self.yamnet_settings ["win_hope_yamnet"] 
        self.win_hope_logmel = self.yamnet_settings ["win_hope_logmel"]
        
        
        self.clip_length_yamnet = int (round(self.clip_length_seconds / self.win_hope_yamnet ))
        self.clip_length_logmel = int (round(self.clip_length_seconds / self.win_hope_logmel))
        
        self.counter = 0
        self.video_name = ''
        self.video_duration = 0

        self.dict_onsets = {}
        self.dict_errors = {}
        self.dict_yamnetoutput = {}
        self.dict_logmel40 = {}
        self.dict_logmel64 = {}
        self.dict_embeddings_yamnet = {}

    def create_video_list (self ):     
        video_dir = os.path.join(self.datadir, 'videos' , self.split) 
        video_recepies = os.listdir(video_dir)
        video_list = []
        for rec in video_recepies:
            files = os.listdir(os.path.join(video_dir, rec))
            video_list.extend([os.path.join(self.split , rec ,f) for f in files])
        return video_list
    
    def load_video (self):       
        video_path = os.path.join(self.datadir, 'videos' , self.video_name) 
        wav_data, sample_rate = librosa.load(video_path , sr=self.target_sr , mono=True)        
        duration = len(wav_data)/self.target_sr
        self.video_duration = duration
        return wav_data
    
    def save_wav_files(self):  
        video_list = self.create_video_list()
        output_path = os.path.join(self.outputdir , 'wavs', self.split )
        os.makedirs(output_path , exist_ok = True)
        self.counter = 0 
        for video_name in video_list:                   
            self.video_name = video_name
            wav_data = self.load_video()
            output_name = output_path  + '/' +   str(self.counter) + '.wav'
            sf.write(output_name, wav_data, self.target_sr)
            self.counter += 1
            
    def load_dict_onsets (self):
        input_name =  os.path.join(self.outputdir, self.exp_name , self.split + '_onsets') 
        with open(input_name, 'rb') as handle:
            dict_onsets = pickle.load(handle)
        return dict_onsets

    def save_wav_clips(self):
        # this function saves audio clips from already detected onset lists
        dict_onsets = self.load_dict_onsets ()
    
        self.counter = 0
        for video_name, value in dict_onsets.items():      
            print(self.counter)
            self.video_name = video_name
            self.folder_name = value['folder_name']
            self.accepted_onsets_second = value['onsets']            
            wav_data = self.load_video()
            output_path = os.path.join(self.outputdir ,  self.exp_name, self.split ,  str(self.folder_name) , "wavs")
            os.makedirs(output_path, exist_ok= True)
            
            for conter_onset, onset in enumerate(self.accepted_onsets_second):
                wav_clip = wav_data[onset*self.target_sr: (onset + self.clip_length_seconds)*self.target_sr]               
                sf.write(output_path + '/' + str(conter_onset) + '.wav' , wav_clip, self.target_sr)
    
    def load_speech_segments (self):
        file_name = self.outputdir + self.split + '_yamnet_speech' 
        with open(file_name, 'rb') as handle:
            speech_segments = pickle.load(handle)
        return speech_segments

    
    def load_logmel_feature (self):
        file_name = self.outputdir + self.split + '_logmels' 
        with open(file_name, 'rb') as handle:
            logmels = pickle.load(handle)
        return logmels

    
    def extract_logmel_features (self, wav_data):
        #print('...now it is extracting logmels step 1...........')
        window_len_in_ms = self.yamnet_settings ['win_length_logmel']
        window_hop_in_ms = self.yamnet_settings ['win_hope_logmel']
        number_of_mel_bands = self.yamnet_settings ['logmel_bands']
        sr_target = self.yamnet_settings ['target_sample_rate' ]
        #print('...now it is extracting logmels step 2...........')
        logmels = utils.calculate_logmels (wav_data , number_of_mel_bands , window_len_in_ms , window_hop_in_ms , sr_target)        
        return logmels

    
    def execute_yamnet (self, wavedata ):  
        scores, embeddings, log_mel_yamnet = self.model(wavedata)       
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


    def produce_onset_candidates_1 (self, speech_segments):              
        clip_length_seconds = self.clip_length_seconds       
        accepted_rate = self.yamnet_settings ["acceptance_snr"]
        skip_seconds = self.yamnet_settings ["skip_seconds"]
        # e.g., 21 frames --> almost equal to ~10 seconds of audio
        # and, for clip of 10 seconds: 21* 0.8 = 17
        # also, skip first 10 seconds of the audio which is 21 yamnet frames
        # sample every 3 seconds (which is ~ 3* 2 yamnet frames)              
        clip_length_yamnet = self.clip_length_yamnet  # e.g. 21 for 10 second      
        accepted_plus = int(round(clip_length_yamnet * accepted_rate)) # e.g. 17
        number_of_clips = int( self.video_duration / clip_length_seconds) 
        
        start_frame = clip_length_yamnet # skip first 10 seconds
        end_frame = len(speech_segments) - clip_length_yamnet
        skip_frames_yamnet = int(round(skip_seconds/ self.win_hope_yamnet))
        initial_sequence = [ onset for onset in range(start_frame , end_frame , skip_frames_yamnet) ]  
        
        trial_index = len(initial_sequence) -1 
        onsets_yamnet = []
        
        upated_sequence = copy.copy(initial_sequence) 
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
       
        onsets_second = [math.floor(item * self.win_hope_yamnet) for item in onsets_yamnet]
        onsets_logmel = [math.floor(item / self.win_hope_logmel) for item in onsets_second]
                       
        print('###############################################################')        
        print(self.counter)
        print(self.video_name)
        print(self.video_duration)
        print(speech_segments)       
        print('###############################################################')
        
        return onsets_yamnet , onsets_second , onsets_logmel 

    def produce_onset_candidates_2 (self, speech_segments):
                          
        accepted_rate = self.yamnet_settings ["acceptance_snr"]
        accepted_overlap_second = self.yamnet_settings ["accepted_overlap_second"]
        
        # e.g., 21 frames --> almost equal to ~10 seconds of audio
        # and, for clip of 10 seconds: 21* 0.8 = 17
        # also, skip first 10 seconds of the audio which is 21 yamnet frames
                     
        clip_length_yamnet = self.clip_length_yamnet  # e.g. 21 for 10 second        
        accepted_plus = int(round(clip_length_yamnet * accepted_rate)) # e.g. 17
                
        # scanning the signal (yamnet output scores)
        scanned_speech = []
        len_silde_window = clip_length_yamnet    
        for counter in range(len(speech_segments)):
            slide_window_temp = speech_segments[counter:counter + len_silde_window]
            speech_portion = sum(slide_window_temp)
            scanned_speech.append(speech_portion)
       
        initial_seq = [onset>= accepted_plus for onset in scanned_speech]
        initial_seq = numpy.multiply(initial_seq,1)
        
        
        # greedy search
        accepted_overlap_yamnet = int(round(accepted_overlap_second/ self.win_hope_yamnet)) #  10 yamnet frames: almost 5 seconds       
        skip_len = clip_length_yamnet - accepted_overlap_yamnet
        
        updated_seq = copy.copy(initial_seq)
        
        for counter, value in enumerate(updated_seq):
            if value==1:
                updated_seq[counter+ 1: counter + skip_len] = 0
                           
        onsets_yamnet = [counter for counter,value in enumerate(updated_seq) if value==1]
        onsets_second = [math.floor(item * self.win_hope_yamnet) for item in onsets_yamnet]
        onsets_logmel = [math.floor(item / self.win_hope_logmel) for item in onsets_second]
    
        print('###############################################################')        
        print(self.counter)
        print(self.video_name)
        print(self.video_duration)
        print(onsets_second)       
        print('###############################################################')
        
        return onsets_yamnet , onsets_second , onsets_logmel     


    def update_yamnet_output (self, speech_segments):
        self.dict_yamnetoutput[self.video_name] = speech_segments

    def update_logmel40_list (self, logmels):
        self.dict_logmel40[self.video_name] = logmels
        
    def update_logmel64_list (self, logmel_yamnet):
        self.dict_logmel64[self.video_name] = logmel_yamnet
        
    def update_embedding_list (self, embeddings_yamnet):
        self.dict_embeddings_yamnet[self.video_name] = embeddings_yamnet
        
    def update_onset_list (self, accepted_onsets_second):      
        self.dict_onsets[self.video_name] = {'onsets': accepted_onsets_second , 'folder_name':self.counter}        

    def update_error_list (self):       
        self.dict_errors[self.counter] = self.video_name       

    def save_yamnet_output (self):
        output_name =  self.outputdir + self.split + '_yamnet_speech' 
        with open(output_name, 'wb') as handle:
            pickle.dump(self.dict_yamnetoutput, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_logmel40 (self):
        output_name =  self.outputdir + self.split + '_logmels40' 
        with open(output_name, 'wb') as handle:
            pickle.dump(self.dict_logmel40, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_logmel64 (self):
        output_name =  self.outputdir + self.split + '_logmels64' 
        with open(output_name, 'wb') as handle:
            pickle.dump(self.dict_logmel64, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def save_embeddings (self):
        output_name =  self.outputdir + self.split + '_embeddings' 
        with open(output_name, 'wb') as handle:
            pickle.dump(self.dict_embeddings_yamnet, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def save_onsets(self):
        output_name =  os.path.join(self.outputdir, self.exp_name , self.split + '_onsets')   
        with open(output_name , 'wb') as handle:
            pickle.dump(self.dict_onsets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def save_error_list (self):
        output_name =  os.path.join(self.outputdir, self.split + '_errors')  
        with open(output_name , 'wb') as handle:
            pickle.dump(self.dict_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    def save_per_video (self, logmel_yamnet, onsets_second , accepted_logmel40, accepted_logmel64, accepted_embeddings):
        output_path = os.path.join(self.outputdir , self.exp_name, self.split ,  str(self.counter))
        os.makedirs(output_path , exist_ok = True)
        output_name = output_path  + '/af'
        dict_out = {}
        dict_out['video_name'] = self.video_name
        dict_out['video_duration'] = self.video_duration
        dict_out['onsets_second'] = onsets_second
        dict_out['logmel40'] = accepted_logmel40
        dict_out['logmel64'] = accepted_logmel64
        dict_out['embeddings'] = accepted_embeddings           
        with open(output_name, 'wb') as handle:
            pickle.dump(dict_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
          
    def find_speech_segments(self, wav_data): 
        scores_yamnet, embeddings_yamnet, logmel_yamnet = self.execute_yamnet(wav_data)
        speech_segments = self.detect_speech_frames(scores_yamnet)
        self.update_yamnet_output(speech_segments)
        return speech_segments, embeddings_yamnet, logmel_yamnet 
    
    def to_run_speech_detection(self):
        video_list = self.create_video_list()
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.counter = 0
        for video_name in video_list:
                       
            self.video_name = video_name
            wav_data = self.load_video()
            logmels = self.extract_logmel_features (wav_data) 
            speech_segments, embeddings_yamnet, logmel_yamnet = self.find_speech_segments(wav_data)

            print('........step 1 is done ...........' )
            self.update_logmel40_list(logmels)
            self.update_logmel64_list(logmel_yamnet)
            self.update_embedding_list(embeddings_yamnet)
            
            onsets_yamnet , onsets_second , onsets_logmel = self.produce_onset_candidates_2 (speech_segments)
            self.update_onset_list (onsets_second)
  
            print('........step 2 is done ...........' )
            accepted_logmel40 = [logmels[onset:onset + self.clip_length_logmel] for onset in onsets_logmel]     
            accepted_logmel64 = [logmel_yamnet[onset:onset + self.clip_length_logmel] for onset in onsets_logmel]
            accepted_embeddings = [embeddings_yamnet[onset:onset + self.clip_length_yamnet] for onset in onsets_yamnet]
            
            self.save_per_video (onsets_second , onsets_logmel, accepted_logmel40, accepted_logmel64,accepted_embeddings )
            
            print('........step 3 is done ...........' )   
                
            self.counter += 1 
        
        self.save_onsets()
        self.save_error_list()
        self.save_yamnet_output()
        self.save_logmel40()
        self.save_logmel64()
        self.save_embeddings()
        
         
        
    def run_from_file (self):    
        dict_speech_segments = self.load_speech_segments()
        dict_logmels = self.load_logmel_feature()       
        video_list = self.create_video_list()
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        for video_name in video_list: 
            
            self.video_name = video_name
            print('video name is .............')
            print(self.video_name)
            print('............')
            try:
                speech_segments = dict_speech_segments[self.video_name]
                print('speech is .............')
                print(len(speech_segments))
                print('............')
                onsets_yamnet , onsets_second , onsets_logmel = self.produce_onset_candidates_2 (speech_segments)
                
                logmels = dict_logmels[self.video_name]
                accepted_logmel40 = [logmels[onset:onset + self.clip_length_logmel] for onset in onsets_logmel] 
                accepted_logmel64 = []
                accepted_embeddings = []
                self.update_onset_list(onsets_second)
                self.save_per_video(onsets_second , onsets_logmel, accepted_logmel40, accepted_logmel64, accepted_embeddings )
                
            except:
                self.update_error_list()
                
            self.counter += 1             
            
        self.save_onsets()
        
        
    def __call__ (self):
        
        if self.run_speech_detection == True:
            self.to_run_speech_detection()
        else:
            self.run_from_file()
            
        

                
 
        

        
        
        
                    
           


            
        
        
        

           
        

        
        
