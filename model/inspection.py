
"""
to inspect hidden layer patterns

"""
import os
import json
import numpy
from train_validate import Train_AVnet
from utils import triplet_loss,  mms_loss,  prepare_data, preparX, preparY, calculate_recallat10 



class Inspection (Train_AVnet):
    def __init__(self,  model_config , training_config):
        Train_AVnet.__init__(self,  model_config , training_config)
            
        self.audio_model_name = model_config["audio_model_name"]
        self.visual_model_name = model_config["visual_model_name"]
        self.visual_layer_name = model_config["visual_layer_name"]      
        #self.loss = model_config["loss"]
        self.zeropadd = model_config["zeropadd_size"]
        
        self.featuredir = training_config["featuredir"]
        self.featuretype = training_config ["featuretype"]  
        self.outputdir = training_config ["outputdir"]
        self.use_pretrained = training_config ["use_pretrained"]
        self.save_results = training_config["save_results"]
        self.plot_results = training_config["plot_results" ]
    
    def load_model_weights(self):
        model_path = os.path.join(self.outputdir, 'model_weights.h5')     
        self.av_model.load_weights(model_path)
        
    def set_audio_model(self):#30 (lambda)
        #self.audio_embedding_model.summary()
        for n in range(32):
            self.audio_embedding_model.layers[n].set_weights(self.av_model.layers[n].get_weights())
        self.audio_embedding_model.layers[36].set_weights(self.av_model.layers[41].get_weights())
        self.audio_embedding_model.layers[37].set_weights(self.av_model.layers[43].get_weights())
    
    def set_visual_model(self):# 38 (lambda)
        #self.visual_embedding_model.summary()#32
        self.visual_embedding_model.layers[4].set_weights(self.av_model.layers[40].get_weights())
        self.visual_embedding_model.layers[5].set_weights(self.av_model.layers[42].get_weights())
   
    def predict_visual_embeddings(self,Ydata):
        visual_embedding = self.visual_embedding_model.predict(Ydata)
        return visual_embedding
    
    def predict_audio_embeddings(self,Xdata):
        audio_embedding = self.audio_embedding_model.predict(Xdata)
        return audio_embedding
 

    def read_json_anns (self):
        dict_anns = {}
        json_path =   "../../data/youcook2/annotations/youcookii_annotations_trainval.json"
        with open(json_path) as handle:
            jsonfile = json.load(handle)
        database = jsonfile['database']
        for video_name, info in database.items():
            if info['subset'] == self.split:
                rcp_type =  info['recipe_type']
                video_fullname = os.path.join(self.split,rcp_type, video_name)
                dict_anns[video_fullname] = info['annotations']
        return dict_anns   
        
    def create_image_list(self):
        list_image_names = []
        
        for video_name, value in self.dict_onsets.items():           
            self.video_name = video_name
            self.folder_name = value['folder_name']   
            accepted_onsets_second = value['onsets']
            accepted_offsets_second = value['offsets']
            number_of_clips = len(accepted_onsets_second)
            number_of_images = numpy.array(accepted_offsets_second) - numpy.array(accepted_onsets_second)
            image_path = os.path.join( self.featuretype , self.split ,  str(self.folder_name) , "images")
            if number_of_clips > 0:
                for counter_clip in range(number_of_clips):  
                    image_subpath = os.path.join(image_path, str(counter_clip)) 
                    list_image_names.append(image_subpath)
                    # image_counts = number_of_images [counter_clip]
                    # for image_counter in range(image_counts):
                    #     image_full_name = os.path.join(image_subpath, str(image_counter))
                    #     #print(image_full_name)
                        
        return list_image_names            
                    
            # if len(value['onsets']) != 0:
            #     self.feature_path = os.path.join(self.featuredir , self.featuretype, self.split ,  str(self.folder_name))  
            #     self.image_path = os.path.join(self.feature_path,"images")
            #     all_images = os.listdir(self.image_path)
            #     dict_data[video_name] = all_images
    def __call__ (self):
        
        [Xshape , Yshape] = self.get_input_shapes()
        self.visual_embedding_model, self.audio_embedding_model, self.av_model = self.build_network( Xshape , Yshape )
        #self.av_model.summary()
        self.featuretype = 'ann-based'
        self.load_model_weights()
        self.set_visual_model()
        self.set_audio_model()
        
        self.split = "validation"       
        self.load_dict_onsets()
        visual_features_test = self.get_visual_features()
        
        audio_features_test = self.get_audio_features() 
        Ydata, Xdata, b_val = prepare_data (audio_features_test , visual_features_test , self.loss, shuffle_data = False) 
        # Y_embedding = self.predict_visual_embeddings(Ydata[0:])
        # X_embedding = self.predict_audio_embeddings(Xdata[0:])
        # list_image_names = self.create_image_list()
        
        dict_anns = self.read_json_anns()
        return dict_anns