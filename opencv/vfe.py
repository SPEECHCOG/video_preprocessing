import os
import numpy
import cv2 as cv
import pickle


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

class VisualFeatureExtractor:
    
    def __init__(self,audio_model,dataset, datadir, outputdir, split , video_settings):
        
        self.audio_model = audio_model
        self.dataset = dataset
        self.datadir = datadir
        self.outputdir = outputdir
        self.split = split
        
        self.video_settings = video_settings       
        self.clip_length_seconds = self.video_settings ["clip_length_seconds"]
         
        self.video_name = ""
        self.output_subpath = ""
       
        self.layer_name = 'block5_conv3' #'fc1'
        
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
    
    def calculate_vgg_features(self,image_fullname):
 
        image_original = cv.imread(image_fullname)
        image_resized = cv.resize(image_original,(224,224))
        image_input_vgg = preprocess_input(image_resized.reshape((1, 224, 224, 3)))
        model = self.model
        vgg_out = model.predict(image_input_vgg)
        vgg_out_reshaped = numpy.squeeze(vgg_out)
        
        return vgg_out_reshaped
    
    
    def save_per_video (self, accepted_onsets_second ,  vgg_video):
 
        output_path = os.path.join(self.outputdir , self.split ,  str(self.folder_name))
        output_name = output_path + "/vf"        
        
        dict_out = {}
        dict_out['video_name'] = self.video_name
        dict_out['onsets_second'] = accepted_onsets_second
        dict_out[self.layer_name] =  vgg_video
                 
        with open(output_name, 'wb') as handle:
            pickle.dump(dict_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def update_error_list (self):
        
        self.dict_errors[self.counter] = self.video_name       
          
    
    def __call__ (self):
        
        currentmodel = VGG16()  
        self.model = Model(inputs=currentmodel.input,outputs=currentmodel.get_layer(self.layer_name).output)
        dict_onsets = self.load_dict_onsets ()
     
        for video_name, value in dict_onsets.items(): 
            self.video_name = video_name
            accepted_onsets_second = value['onsets']
            self.folder_name = value['folder_name']
            vgg_video =  []
            for counter_onset, onset in accepted_onsets_second:             
                image_subpath , image_names = self.create_image_list (counter_onset)
                all_vggs_per_onset = []
                for name in image_names: 
                    image_fullname = image_subpath +  "/" + name
                    vgg_image = self.calculate_vgg_features(image_fullname)
                    all_vggs_per_onset.append(vgg_image)
                    
                vgg_video.append(numpy.array(all_vggs_per_onset)) 
             
             
            self.save_per_video ( accepted_onsets_second ,  vgg_video)
            # try:                          
            # except:
            #     self.update_error_list()
       


        # output_name =  self.outputdir + self.split + '_image_errors'  
        # with open(output_name , 'wb') as handle:
        #     pickle.dump(self.dict_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)