import os
import numpy
import cv2 as cv
import pickle


class Analysis:
    
    def __init__(self,audio_model,dataset, datadir, outputdir, split , video_settings):
        
        self.audio_model = audio_model
        self.dataset = dataset
        self.datadir = datadir
        self.outputdir = outputdir
        self.split = split
        
        self.video_settings = video_settings
        
        self.clip_length_seconds = self.video_settings ["clip_length_seconds"]
        
        self.folder_counter = 0
        self.video_name = ""
        self.video_duration = 0
        self.output_subpath = ""
       
        self.dict_errors = {}
        
    def create_video_list (self ):
        
        video_dir = os.path.join(self.datadir, self.split)
        video_list = os.listdir(video_dir)
        return video_list
 
    
    def load_video (self):
        
        video_name = self.video_name
        video_sample = os.path.join(self.datadir,self.split, video_name) 
        
        #video_sample = "../data/input/101_YSes0R7EksY.mp4"
        cap = cv.VideoCapture(video_sample)
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        return cap


        
    def write_clip_images (self, cap , accepted_onsets_second , accepted_offsets_second):
        
        output_path = os.path.join(self.outputdir , self.split ,  str(self.folder_counter) , "images")
        os.mkdir(output_path)      
        number_of_clips = len(accepted_onsets_second)
        for counter_clip in range(number_of_clips):
            
            self.output_subpath = os.path.join(output_path, str(counter_clip))
            os.mkdir(self.output_subpath)
            onset = accepted_onsets_second [counter_clip]
            offset = accepted_offsets_second [counter_clip]
            all_seconds = numpy.arange(onset,offset)
            for counter_second, second in enumerate(all_seconds):

                output_name = self.output_subpath  +  "/" + str(counter_second) + ".jpg"
                print(output_name)
                ms =  second * 1000
                cap.set(cv.CAP_PROP_POS_MSEC, ms)      
                ret,frame = cap.read()                   
                cv.imwrite(output_name, frame)                      
 
    
    def load_dict_onsets (self):
        
        input_name =  self.outputdir + self.split + '_onsets'  
        with open(input_name, 'rb') as handle:
            dict_onsets = pickle.load(handle)
        return dict_onsets

    def update_error_list (self):
        
        self.dict_errors[self.counter] = self.video_name       
          
        
    
    def __call__ (self):
        
        # video_list = self.create_video_list()
        dict_onsets = self.load_dict_onsets ()
        self.counter = 0
        for video_name, value in dict_onsets.items(): 
            
            print(self.counter)
            self.video_name = video_name
            accepted_onsets_second = value['onsets']
            accepted_offsets_second = value['offsets']
            self.folder_counter = value['folder_name']
            cap = self.load_video ()
            self.write_clip_images(cap, accepted_onsets_second, accepted_offsets_second)
            # try:           
                
            # except:
            #     self.update_error_list()
            self.counter += 1


        # output_name =  self.outputdir + self.split + '_image_errors'  
        # with open(output_name , 'wb') as handle:
        #     pickle.dump(self.dict_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)