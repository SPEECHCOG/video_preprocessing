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



class VisualFeatureExtractor:
    
    def __init__(self,visual_model_name, layer_name, dataset, datadir, outputdir, split , video_settings):
        self.dataset = dataset
        self.datadir = datadir
        self.outputdir = outputdir
        self.split = split
        
        self.video_settings = video_settings       
        self.clip_length_seconds = self.video_settings ["clip_length_seconds"]
        self.visual_model_name = visual_model_name
        self.layer_name = layer_name
         
        self.video_name = ""  
        self.dict_errors = {}
    
    def load_dict_onsets (self):  
        input_name =  self.outputdir + self.split + '_onsets'  
        with open(input_name, 'rb') as handle:
            dict_onsets = pickle.load(handle)
        return dict_onsets
        
    def create_image_list (self , counter_onset):
        image_path = os.path.join(self.outputdir , self.split ,  str(self.folder_name) , "images")
        image_subpath = os.path.join(image_path, str(counter_onset))        
        image_names = os.listdir(image_subpath)
       
        return image_subpath , image_names
     
    def load_model (self):    
        if self.visual_model_name == "vgg16":
            base_model = VGG16()
        if self.visual_model_name == "resnet152":
            base_model = ResNet152(weights='imagenet')
        if self.visual_model_name == "Xception":
            base_model = Xception(weights='imagenet')
        return base_model
        
    def prepare_image (self, image_fullname):
        image_original = image.load_img(image_fullname, target_size=(224, 224))
        image_array = image.img_to_array(image_original)
        image_dim = numpy.expand_dims(image_array, axis=0)
        if self.visual_model_name == "vgg16":
            image_input = preprocess_input_vgg(image_dim)
        if self.visual_model_name =="resnet152":
            image_input = processing_input_resnet(image_dim)
        if self.visual_model_name =="Xception":
            image_input = processing_input_xception(image_dim) 
        return image_input
    
    def calculate_visual_features(self, image_fullname):
        image_input = self.prepare_image (image_fullname)   
        features_out = self.visual_model.predict(image_input)
        features_out_reshaped = numpy.squeeze(features_out)
        return   features_out_reshaped     

          
    def save_per_video (self, accepted_onsets_second ,  vfs_video):
        output_path = os.path.join(self.outputdir , self.split ,  str(self.folder_name))
        output_name = output_path + "/vf_" + self.visual_model_name                
        dict_out = {}
        dict_out['video_name'] = self.video_name
        dict_out['onsets_second'] = accepted_onsets_second
        dict_out[self.visual_model_name + '_' + self.layer_name] =  vfs_video
                   
        with open(output_name, 'wb') as handle:
            pickle.dump(dict_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def update_error_list (self):
        
        self.dict_errors[self.folder_name] = self.video_name       
          
    
    def __call__ (self):
        
        base_model = self.load_model()
        
        self.visual_model = Model(inputs=base_model.input,outputs=base_model.get_layer(self.layer_name).output)
        dict_onsets = self.load_dict_onsets ()
     
        for video_name, value in dict_onsets.items(): 
            self.video_name = video_name # e.g. "testing/101/YSes0R7EksY.mp4"
            accepted_onsets_second = value['onsets']
            self.folder_name = value['folder_name']
            print ("processing video folder ... " + str(self.folder_name) )
            vf_video =  []
            number_of_clips = len(accepted_onsets_second)
            for counter_clip in range(number_of_clips):
                try:      
                    image_subpath , image_names = self.create_image_list (counter_clip)
                    all_vfs_per_onset = []
                    for name in image_names: 
                        image_fullname = os.path.join(image_subpath , name)
                        vf_image = self.calculate_visual_features(image_fullname)
                        all_vfs_per_onset.append(vf_image)
                        
                    vf_video.append(numpy.array(all_vfs_per_onset)) 
                except:
                    self.update_error_list()
            self.save_per_video( accepted_onsets_second ,  vf_video)


        output_name =  self.outputdir + self.split + '_image_errors'  
        with open(output_name , 'wb') as handle:
            pickle.dump(self.dict_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)