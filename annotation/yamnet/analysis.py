import os
import librosa
import math
import pickle
import json
import soundfile as sf
import utils
import tensorflow_hub as hub
 
class Analysis:
    
    def __init__(self,audio_model, dataset, datadir, outputdir, split , yamnet_settings):
        
        self.audio_model = audio_model
        self.dataset = dataset
        self.datadir = datadir
        self.outputdir = outputdir
        self.split = split
        self.yamnet_settings = yamnet_settings
        self.target_sr = self.yamnet_settings ["target_sample_rate"] 
        self.annfile_trainval = os.path.join(self.datadir, 'annotations' , 'youcookii_annotations_trainval.json')
        self.annfile_test = os.path.join(self.datadir, 'annotations' , 'youcookii_annotations_test_segments_only.json')
               
        self.win_hope_yamnet = self.yamnet_settings ["win_hope_yamnet"] 
        self.win_hope_logmel = self.yamnet_settings ["win_hope_logmel"]
       
        self.annfile = ''
        self.database = {}
        self.counter = 0
        self.video_name = ''
        self.video_duration = 0

        self.dict_onsets = {}
        self.dict_errors = {}
        
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

    def load_dict_onsets (self):
        input_name =  os.path.join(self.outputdir , self.split + '_onsets') 
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
            self.accepted_offsets_second = value['offsets']
            wav_data = self.load_video()
            output_path = os.path.join(self.outputdir ,  self.split ,  str(self.folder_name) , "wavs")
            os.makedirs(output_path, exist_ok= True)
            
            for counter_onset, onset in enumerate(self.accepted_onsets_second):
                offset = self.accepted_offsets_second [counter_onset]
                wav_clip = wav_data[onset*self.target_sr: offset*self.target_sr]               
                sf.write(output_path + '/' + str(counter_onset) + '.wav' , wav_clip, self.target_sr)
    
    def extract_logmel_features (self, wav_data):
        
        window_len_in_ms = self.yamnet_settings ['win_length_logmel']
        window_hop_in_ms = self.yamnet_settings ['win_hope_logmel']
        number_of_mel_bands = self.yamnet_settings ['logmel_bands']
        sr_target = self.yamnet_settings ['target_sample_rate' ]
        logmels = utils.calculate_logmels (wav_data , number_of_mel_bands , window_len_in_ms , window_hop_in_ms , sr_target)        
        return logmels
    
    
    def execute_yamnet (self, wavedata): 
               
        scores, embeddings, log_mel_yamnet = self.model(wavedata)       
        scores_np = scores.numpy()
        logmel_yamnet_np = log_mel_yamnet.numpy()
        embeddings_np = embeddings.numpy()
        
        #max_scores = scores_np.max(axis = 1)
        max_class_indexes = scores_np.argmax(axis = 1)
        cns = self.class_names
        max_class_names = [cns[ind] for ind in max_class_indexes]

        return max_class_indexes , max_class_names, embeddings_np, logmel_yamnet_np
 
    
    def read_annotations (self): 
        if self.split == "testing":
            self.annfile = self.annfile_test
        else:
            self.annfile = self.annfile_trainval
        
        with open(self.annfile, 'rb') as handle:
             output = json.load(handle)
        
        self.database = output['database']
        
    
    def update_onset_list (self):        
        self.dict_onsets[self.video_name] = {'onsets': self.onsets_second , 'offsets':self.offsets_second, 'folder_name':self.counter}

    def update_error_list (self):
        
        self.dict_errors[self.counter] = self.video_name       
    
    def save_per_video (self, max_class_indexes , max_class_names , embeddings, logmel_yamnet, logmels):
        accepted_yamnetlogmels = [ logmel_yamnet[self.onsets_logmel[counter]:self.offsets_logmel[counter]] for counter in range(len(self.onsets_logmel))]
        accepted_logmels = [ logmels[self.onsets_logmel[counter]:self.offsets_logmel[counter]] for counter in range(len(self.onsets_logmel))] 
        accepted_embeddings = [embeddings[self.onsets_yamnet[counter]:self.offsets_yamnet[counter]] for counter in range(len(self.onsets_yamnet))]
        output_path = os.path.join(self.outputdir , self.split ,  str(self.counter))
        os.makedirs(output_path, exist_ok = True)
        output_name = output_path  + '/af'
        dict_out = {}
        dict_out['video_name'] = self.video_name
        dict_out['video_duration'] = self.video_duration
        dict_out['onsets_second'] = self.onsets_second
        dict_out['offsets_second'] = self.offsets_second
        dict_out['embeddings'] = accepted_embeddings
        dict_out['logmel40'] = accepted_logmels
        dict_out['logmel64'] = accepted_yamnetlogmels
        dict_out['class_names'] = max_class_names
        dict_out['class_indexes'] = max_class_indexes
              
        with open(output_name, 'wb') as handle:
            pickle.dump(dict_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
            
    def read_ann_segments (self):
        fullname = self.video_name
        name_with_extension = fullname.split('/')[-1]       
        name = name_with_extension.split('.') [0]
        ann_video = self.database[name]
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
            
        self. onsets_second = onsets_second
        self. offsets_second = offsets_second
        self.onsets_logmel = [math.floor(item / self.win_hope_logmel) for item in onsets_second]
        self.offsets_logmel = [math.floor(item / self.win_hope_logmel) for item in offsets_second]
        
        self.onsets_yamnet = [math.floor(item / self.win_hope_yamnet) for item in onsets_second]
        self.offsets_yamnet = [math.floor(item / self.win_hope_yamnet) for item in offsets_second]
            
    
    def __call__ (self):
        
        video_list = self.create_video_list()
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.class_names = self.read_yamnet_classes()
        self.read_annotations()
        for video_name in video_list:         
            self.video_name = video_name
            # do all analysis
            try:
                wav_data = self.load_video ()
                logmels = self.extract_logmel_features (wav_data)
                max_class_indexes , max_class_names, embeddings, logmel_yamnet = self.execute_yamnet(wav_data)                
                self.read_ann_segments ()
                
                self.update_onset_list ()
                self.save_per_video ( max_class_indexes , max_class_names , embeddings, logmel_yamnet, logmels)    
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