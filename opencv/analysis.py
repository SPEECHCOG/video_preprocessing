import os
import numpy
import cv2 as cv
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg

from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input as processing_input_resnet

from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input as processing_input_xception


class Analysis:
    
    def __init__(self,visual_model_name, visual_layer_name , save_images, save_visual_features, datadir, outputdir, exp_name, split, video_settings):
        
        
        self.visual_model_name = visual_model_name
        self.visual_layer_name = visual_layer_name
        self.save_images = save_images
        self.save_visual_features = save_visual_features
     
        self.datadir = datadir
        self.outputdir = outputdir
        self.split = split
        self.exp_name = exp_name
        
        self.video_settings = video_settings
        
        self.clip_length_seconds = self.video_settings ["clip_length_seconds"]
        
        self.video_name = ""
        self.video_duration = 0
        self.output_subpath = ""
       
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
        # e.g. "testing/101/YSes0R7EksY.mp4"
        video_sample = os.path.join(self.datadir, 'videos' , self.video_name)       
        cap = cv.VideoCapture(video_sample)
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        return cap

        
    # def create_video_list (self ):
        
    #     video_dir = os.path.join(self.datadir, self.split)
    #     video_list = os.listdir(video_dir)
    #     return video_list
 
    
    # def load_video (self):
        
    #     video_name = self.video_name
    #     video_sample = os.path.join(self.datadir,self.split, video_name) 
        
    #     #video_sample = "../data/input/101_YSes0R7EksY.mp4"
    #     cap = cv.VideoCapture(video_sample)
    #     if (cap.isOpened()== False):
    #         print("Error opening video stream or file")
    #     return cap
    def load_model (self):    
        if self.visual_model_name == "vgg16":
            base_model = VGG16()
        if self.visual_model_name == "resnet152":
            base_model = ResNet152(weights='imagenet')
        if self.visual_model_name == "Xception":
            base_model = Xception(weights='imagenet')
        return base_model
        
    def prepare_image (self, image_original):
        
        image_array = image.img_to_array(image_original)
        image_dim = numpy.expand_dims(image_array, axis=0)
        if self.visual_model_name == "vgg16":
            image_input = preprocess_input_vgg(image_dim)
        if self.visual_model_name =="resnet152":
            image_input = processing_input_resnet(image_dim)
        if self.visual_model_name =="Xception":
            image_input = processing_input_xception(image_dim) 
        return image_input
    
    def calculate_visual_features(self, image_original):
        image_input = self.prepare_image (image_original)   
        features_out = self.visual_model.predict(image_input)
        features_out_reshaped = numpy.squeeze(features_out)
        return   features_out_reshaped     

         

    def save_per_video (self,  vfs_video):
        output_path = os.path.join(self.outputdir , self.exp_name , self.split ,  str(self.folder_name))
        output_name = output_path + "/vf_" + self.visual_model_name                
        dict_out = {}
        dict_out['video_name'] = self.video_name
        dict_out['onsets_second'] = self.accepted_onsets_second
        dict_out[self.visual_model_name + '_' + self.visual_layer_name] =  vfs_video
                   
        with open(output_name, 'wb') as handle:
            pickle.dump(dict_out, handle, protocol=pickle.HIGHEST_PROTOCOL)        
        
    def find_video_features(self, cap): 
         # output_path = os.path.join(self.outputdir ,  self.exp_name, self.split ,  str(self.folder_name) , "images")
         # os.makedirs(output_path, exist_ok= True)
         vf_video =  []
         for counter_onset, onset in enumerate(self.accepted_onsets_second):
            all_vfs_per_onset = []
            for i in range(self.clip_length_seconds):
                ms =  (onset + i ) * 1000
                cap.set(cv.CAP_PROP_POS_MSEC, ms)      
                ret,frame = cap.read() 
                if ret:
                    #image_original = image.load_img(image_fullname, target_size=(224, 224))
                    image_original = cv.resize(frame,(224, 224) )
                    features_out_reshaped = self.calculate_visual_features(image_original)
                    all_vfs_per_onset.append(features_out_reshaped) 
            vf_video.append(all_vfs_per_onset)       
         return vf_video
     
    def write_clip_images (self, cap ):
        
        output_path = os.path.join(self.outputdir ,  self.exp_name, self.split ,  str(self.folder_name) , "images")
        os.makedirs(output_path, exist_ok= True)      
        
        for counter_onset, onset in enumerate(self.accepted_onsets_second):
            
            self.output_subpath = os.path.join(output_path, str(counter_onset))
            os.makedirs(self.output_subpath, exist_ok= True)
            for i in range(self.clip_length_seconds):

                output_name = self.output_subpath  +  "/" + str(i) + ".jpg"
                print(output_name)
                ms =  (onset + i ) * 1000
                cap.set(cv.CAP_PROP_POS_MSEC, ms)      
                ret,frame = cap.read() 
                if ret:                   
                    cv.imwrite(output_name, frame)                      
 
    
    def load_dict_onsets (self):
        input_name =  os.path.join(self.outputdir, self.exp_name , self.split + '_onsets') 
        with open(input_name, 'rb') as handle:
            dict_onsets = pickle.load(handle)
        return dict_onsets

    def update_error_list (self):       
        self.dict_errors[self.counter] = self.video_name       

  
        
    def to_save_visual_features(self):
        base_model = self.load_model()      
        self.visual_model = Model(inputs=base_model.input,outputs=base_model.get_layer(self.visual_layer_name).output)
        
        dict_onsets = self.load_dict_onsets ()
        self.counter = 0        
        for video_name, value in dict_onsets.items(): 
            self.video_name = video_name
            self.folder_name = value['folder_name']
            self.accepted_onsets_second = value['onsets']
            
            cap = self.load_video ()
            vf_video = self.find_video_features(cap,)
            self.save_per_video(vf_video)
            self.counter += 1
            
            
    def to_save_images (self):
        # video_list = self.create_video_list()
        dict_onsets = self.load_dict_onsets ()
        
        for video_name, value in dict_onsets.items():      
            print(self.counter)
            self.video_name = video_name
            self.folder_name = value['folder_name']
            self.accepted_onsets_second = value['onsets']            
            cap = self.load_video ()
            
            self.write_clip_images(cap)
            
            
            
    def __call__ ( self):
        
        if self.save_visual_features:
            base_model = self.load_model()      
            self.visual_model = Model(inputs=base_model.input,outputs=base_model.get_layer(self.visual_layer_name).output)
        
        dict_onsets = self.load_dict_onsets ()
        #self.counter = 0        
        for video_name, value in dict_onsets.items(): 
            self.video_name = video_name
            self.folder_name = value['folder_name']
            
            self.accepted_onsets_second = value['onsets']            
            cap = self.load_video ()
            if self.save_visual_features:
                vf_video = self.find_video_features(cap)
                self.save_per_video( vf_video)
            if self.save_images:
                self.write_clip_images(cap)
                
            #self.counter += 1 
            print(self.folder_name)

