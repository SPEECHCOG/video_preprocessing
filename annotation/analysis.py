import os
import librosa
import numpy
from random import shuffle
import math
import pickle
import json

import utils

import tensorflow_hub as hub

class Analysis:
    
    def __init__(self,audio_model,dataset, datadir, outputdir, split ,annfile_trainval,annfile_test, yamnet_settings):
        
        self.audio_model = audio_model
        self.dataset = dataset
        self.datadir = datadir
        self.outputdir = outputdir
        self.split = split
        self.annfile_trainval = annfile_trainval
        self.annfile_test = annfile_test
        
        self.yamnet_settings = yamnet_settings
        
        self.clip_length_seconds = self.yamnet_settings ["clip_length_seconds"]
        self.win_hope_yamnet = self.yamnet_settings ["win_hope_yamnet"] 
        self.win_hope_logmel = self.yamnet_settings ["win_hope_logmel"]
        
        self.clip_length_yamnet = int (round(self.clip_length_seconds / self.win_hope_yamnet ))
        self.clip_length_logmel = int (round(self.clip_length_seconds / self.win_hope_logmel))
        
        self.annfile = ''
        self.database = {}
        self.counter = 0
        self.video_name = ''
        self.video_duration = 0

        self.dict_onsets = {}
        self.dict_errors = {}
        
    def create_video_list (self ):
        
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

    
    def read_yamnet_classes(self):
        infile = os.path.join(self.outputdir,'yamnet_classes')
        with open(infile, 'rb') as handle:
            class_names = pickle.load(handle)
        return class_names 

    
    def extract_logmel_features (self, wav_data):
        
        window_len_in_ms = self.yamnet_settings ['win_length_logmel']
        window_hop_in_ms = self.yamnet_settings ['win_hope_logmel']
        number_of_mel_bands = self.yamnet_settings ['logmel_bands']
        sr_target = self.yamnet_settings ['target_sample_rate' ]
        logmels = utils.calculate_logmels (wav_data , number_of_mel_bands , window_len_in_ms , window_hop_in_ms , sr_target)        
        return logmels
    
    
    def execute_yamnet (self, wavedata , class_names ): 
               
        scores, embeddings, log_mel_yamnet = self.model(wavedata)       
        scores_np = scores.numpy()
        logmel_yamnet_np = log_mel_yamnet.numpy()
        #embeddings_np = embeddings.numpy()
        
        #max_scores = scores_np.max(axis = 1)
        max_class_indexes = scores_np.argmax(axis = 1)
        max_class_names = [class_names[ind] for ind in max_class_indexes]

        return max_class_indexes , max_class_names, logmel_yamnet_np
 
    
    def read_annotations (self): 
        if self.split == "test":
            self.annfile = self.annfile_test
        else:
            self.annfile = self.annfile_trainval
        
        with open(self.annfile, 'rb') as handle:
             output = json.load(handle)
        
        self.database = output['database']
        
    
    def update_onset_list (self, onsets_second , offsets_second):
        
        self.dict_onsets[self.video_name] = {'onsets': onsets_second , 'offsets':offsets_second, 'folder_name':self.counter}

    def update_error_list (self):
        
        self.dict_errors[self.counter] = self.video_name       
    
    def save_per_video (self, max_class_indexes , max_class_names ,logmel_yamnet, logmels , onsets_second , offsets_second,  onsets_logmel , offsets_logmel):
 

        accepted_yamnetlogmels = [ logmel_yamnet[onsets_logmel[counter]:offsets_logmel[counter]] for counter in range(len(onsets_logmel))]
        accepted_logmels = [ logmels[onsets_logmel[counter]:offsets_logmel[counter]] for counter in range(len(onsets_logmel))] 
        
        output_path = os.path.join(self.outputdir , self.split ,  str(self.counter))
        os.mkdir(output_path)
        output_name = output_path  + '/af'
        dict_out = {}
        dict_out['video_name'] = self.video_name
        dict_out['video_duration'] = self.video_duration
        dict_out['onsets_second'] = onsets_second
        dict_out['offsets_second'] = offsets_second
        dict_out['logmel40'] = accepted_logmels
        dict_out['logmel64'] = accepted_yamnetlogmels
        dict_out['class_names'] = max_class_names
        dict_out['class_indexes'] = max_class_indexes
              
        with open(output_name, 'wb') as handle:
            pickle.dump(dict_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

            
    def read_ann_segments (self):
        fullname = self.video_name
        name_splitted = os.path.splitext(fullname)
        name = name_splitted [0] 
        ann_video = self.database[name[-11:]]
        ("check if durations are matching..........")
        print(ann_video ['duration'])
        print(self.video_duration)
        (".........................................")
        annotations = ann_video['annotations']
        onsets_second = [] 
        offsets_second = []
        for element in annotations:
            segment = element['segment']
            onsets_second.append(segment[0])
            offsets_second.append(segment[1])
            
       
        onsets_logmel = [math.floor(item / self.win_hope_logmel) for item in onsets_second]
        offsets_logmel = [math.floor(item / self.win_hope_logmel) for item in offsets_second]
        return onsets_second , offsets_second, onsets_logmel , offsets_logmel    

    
    def __call__ (self):
        
        video_list = self.create_video_list()
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        class_names = self.read_yamnet_classes()
        self.read_annotations()
        for video_name in video_list:         
            self.video_name = video_name
            # do all analysis
            try:
                wav_data = self.load_video ()
                logmels = self.extract_logmel_features (wav_data)
                max_class_indexes , max_class_names, logmel_yamnet = self.execute_yamnet(wav_data,class_names)                
                onsets_second , offsets_second, onsets_logmel , offsets_logmel = self.read_ann_segments ()
                
                self.update_onset_list (onsets_second , offsets_second)
                self.save_per_video ( max_class_indexes , max_class_names , logmel_yamnet, logmels ,  onsets_second , offsets_second,  onsets_logmel , offsets_logmel)    
            except:
                self.update_error_list()
                
            self.counter += 1
            print(self.counter)
            
        output_name =  self.outputdir + self.split + '_onsets'  
        with open(output_name , 'wb') as handle:
            pickle.dump(self.dict_onsets, handle, protocol=pickle.HIGHEST_PROTOCOL)

        output_name =  self.outputdir + self.split + '_errors'  
        with open(output_name , 'wb') as handle:
            pickle.dump(self.dict_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)