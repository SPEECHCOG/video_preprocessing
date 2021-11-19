import os
#import pickle  # for video env
import pickle5 as pickle # for myPython env
import numpy


from utils import  preparX, preparY

output_dir_features = "/worktmp/khorrami/project_5/video/data/youcook2/features/yamnet/zp50/vf_val"



class Features():
    def __init__(self, audiochannel, visual_model, layer_name, featuredir_train, featuredir_test, outputdir, split , feature_settings):
        self.audiochannel = audiochannel
        self.visual_model = visual_model
        self.featuredir_train = featuredir_train
        self.featuredir_test = featuredir_test
        self.outputdir = outputdir
        self.split = split
        
        self.feature_settings = feature_settings     
        self.clip_length_seconds = self.feature_settings ["clip_length_seconds"]
        self.zeropadd = self.feature_settings["zeropadd"]
        self.visual_model = visual_model
        self.layer_name = layer_name
         
        self.video_name = ''
        self.folder_name = ''
        self.feature_path = '' 
        self.featuredir = self.featuredir_test
        self.dict_errors = {}
        
    def load_dict_onsets (self):       
        input_name =  self.featuredir + self.split + '_onsets'  
        with open(input_name, 'rb') as handle:
            dict_onsets = pickle.load(handle)
        return dict_onsets        


    def load_af (self):       
        af_file = os.path.join(self.feature_path , 'af')   
        with open(af_file, 'rb') as handle:
            af = pickle.load(handle)           
        return af    

    def load_vf (self):       
        vf_file = os.path.join(self.feature_path , 'vf_' + self.visual_model)   
        with open(vf_file, 'rb') as handle:
            vf = pickle.load(handle)           
        return vf   
    
    def update_error_list (self):
        self.dict_errors[self.video_name] = self.folder_name
          
    
    def get_audio_features (self , split):
        if split == "test":
            self.featuredir = self.featuredir_test 
        else:
            self.featuredir = self.featuredir_train
        dict_onsets = self.load_dict_onsets ()  
        af_all = []                
        for video_name, value in dict_onsets.items():             
            self.video_name = video_name
            self.folder_name = value['folder_name']            
            if len(value['onsets']) == 0:
                self.update_error_list()
            else:
                self.feature_path = os.path.join(self.featuredir , self.split ,  str(self.folder_name))      
                af = self.load_af()            
                logmel_all = af['logmel40'] 
                logmel = logmel_all#[0:10]
                if self.zeropadd > 0:
                    len_of_longest_sequence = 1000 * self.zeropadd
                    logmel_padded = preparX (logmel,len_of_longest_sequence )
                    af_all.extend(logmel_padded)
                    #audio_features = numpy.array(af_all)
                    # if normalization of logmels
                    audio_features = normalizeX (af_all, 5000)
                else:
                    af_all.extend(logmel)
                    #audio_features = numpy.array(af_all)
                    # if normalization of logmels
                    audio_features = normalizeX (af_all, 1000)
        
        return audio_features
    
    def get_visual_features (self , split):
        if split == "test":
            self.featuredir = self.featuredir_test 
        else:
            self.featuredir = self.featuredir_train
        dict_onsets = self.load_dict_onsets ()
       
        vf_all = []       
        for video_name, value in dict_onsets.items():             
            self.video_name = video_name
            self.folder_name = value['folder_name']                                  
            self.feature_path = os.path.join(self.featuredir , self.split ,  str(self.folder_name))      
            vf = self.load_vf()
            resnet_all = vf['resnet152_avg_pool'] # resnet features for each onset (10*2048)
            resnet = resnet_all #[0:10] 
            if self.zeropadd > 0:
                len_of_longest_sequence = self.zeropadd
                resnet_padded = preparY (resnet , len_of_longest_sequence) # 50*2048
                vf_all.extend(resnet_padded)
            else:
                vf_all.extend(resnet) 
                
        visual_features = numpy.array(vf_all)
        return visual_features  
    
    def save_features(self, features , output_dir_features):
        
        with open(output_dir_features, 'wb') as handle:
            pickle.dump(features , handle,protocol=pickle.HIGHEST_PROTOCOL)
       
    def __call__ (self):
        
        
        print('.............. getting test features ...............')
        self.split = "test"
        audio_features_test = self.get_audio_features(self.split)            
        visual_features_test = self.get_visual_features(self.split)
        