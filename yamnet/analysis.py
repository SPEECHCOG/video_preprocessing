
import os

class Analysis:
    
    def __init__(self,audio_model,dataset, datadir,outputdir,split , yamnet_settings):
        
        self.audio_model = audio_model
        self.dataset = dataset
        self.datadir = datadir
        self.outputdir = outputdir
        self.split = split
        
        self.yamnet_settings = yamnet_settings
        
    def create_video_list (self ):
        # use npath and take name of all videos
        # return video name list
        listdir = os.listdir(self.datadir)
        return listdir
    
    def load_video (self):
        # use librosa to load one given video
        pass
    
    def resample_audio (self):
        #put this either here or in utils
        pass
    
    def extract_logmel_features (self):
        #use given parameters to extract logmel features for one given audio file        
        pass
    
    def execute_yamnet (self):
        # load model and find speech segments
        # get model output + spectogram + embeddings
        pass
    
    def detect_speech_frames (self):
        # output yamnet as 0/1 labels
        pass
    
    def produce_onset_candidates (self):
        # use 0.8 to produce onset candidates for clips of len 10 seconds
        pass
    
    def convert_frame_to_seconds (self):
        # either here or in utils
        pass
    
    def convert_second_to_frame (self):
        # either here or in utils
        pass
    
    def update_onset_list (self):
        # add results after each video
        pass
    
    def save_per_video (self):
        # save features and onsets for each video in video path
        pass
        
    
    def __call__ (self):
        # call above functions one by one
        print(self.create_video_list())
        pass