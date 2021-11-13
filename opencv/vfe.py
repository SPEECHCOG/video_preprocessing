import os
import numpy
import cv2 as cv
from tensorflow.keras.preprocessing import image
import pickle

from tensorflow.keras.models import Model


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg

from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input as processing_input_resnet

class VisualFeatureExtractor:
    
    def __init__(self,visual_model_name, visual_layer_name, dataset, datadir, outputdir, split , video_settings):

        self.dataset = dataset
        self.datadir = datadir
        self.outputdir = outputdir
        self.split = split
        
        self.video_settings = video_settings       
        self.clip_length_seconds = self.video_settings ["clip_length_seconds"]
        self.visual_model_name = visual_model_name
        self.visual_layer_name = visual_layer_name
         
        self.video_name = ""
        #self.output_subpath = ""    
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
        
        visual_model = self.visual_model
        if visual_model == "vgg16":
            base_model = VGG16()
        if visual_model == "resnet152":
            base_model = ResNet152(weights='imagenet')
        return base_model
    
    def calculate_vgg_features(self,image_fullname):
        
        model = self.model
        image_original = image.load_img(image_fullname, target_size=(224, 224))
        image_array = image.img_to_array(image_original)
        image_dim = numpy.expand_dims(image_array, axis=0)
        image_input_vgg = preprocess_input_vgg(image_dim)
        vgg_out = model.predict(image_input_vgg)
        features_out_reshaped = numpy.squeeze(vgg_out)      
        return features_out_reshaped


    def calculate_resnet_features(self,image_fullname):
        
        model = self.model      
        image_original = image.load_img(image_fullname, target_size=(224, 224))
        image_array = image.img_to_array(image_original)
        image_dim = numpy.expand_dims(image_array, axis=0)
        image_input_resnet = processing_input_resnet(image_dim)     
        resnet_out = model.predict(image_input_resnet)
        features_out_reshaped = numpy.squeeze(resnet_out)     
        return features_out_reshaped    
 
    def calculate_image_features (self,image_fullname):
        if self.visual_model == "vgg16":
            features_out_reshaped = self.calculate_vgg_features(image_fullname)
        if self.visual_model =="resnet152":
            features_out_reshaped = self.calculate_resnet_features(image_fullname)
        return   features_out_reshaped  
            
           
    def save_per_video (self, accepted_onsets_second ,  vgg_video):
 
        output_path = os.path.join(self.outputdir , self.split ,  str(self.folder_name))
        output_name = output_path + "/vf_" + self.visual_model                
        dict_out = {}
        dict_out['video_name'] = self.video_name
        dict_out['onsets_second'] = accepted_onsets_second
        dict_out[self.visual_model + '_' + self.layer_name] =  vgg_video
                   
        with open(output_name, 'wb') as handle:
            pickle.dump(dict_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def update_error_list (self):
        
        self.dict_errors[self.counter] = self.video_name       
          
    
    def __call__ (self):
        
        base_model = self.load_model()
        
        self.model = Model(inputs=base_model.input,outputs=base_model.get_layer(self.layer_name).output)
        dict_onsets = self.load_dict_onsets ()
     
        for video_name, value in dict_onsets.items(): 
            self.video_name = video_name
            accepted_onsets_second = value['onsets']
            self.folder_name = value['folder_name']
            print ("processing video folder ... " + str(self.folder_name) )
            vf_video =  []
            for counter_onset, onset in enumerate(accepted_onsets_second):             
                image_subpath , image_names = self.create_image_list (counter_onset)
                all_vfs_per_onset = []
                for name in image_names: 
                    image_fullname = os.path.join(image_subpath , name)
                    vf_image = self.calculate_image_features(image_fullname)
                    all_vfs_per_onset.append(vf_image)
                    
                vf_video.append(numpy.array(all_vfs_per_onset)) 

            self.save_per_video ( accepted_onsets_second ,  vf_video)
            # try:                          
            # except:
            #     self.update_error_list()
       


        # output_name =  self.outputdir + self.split + '_image_errors'  
        # with open(output_name , 'wb') as handle:
        #     pickle.dump(self.dict_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)