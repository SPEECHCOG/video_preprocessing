
from model import AVnet

from utils import triplet_loss,  mms_loss,  prepare_data, preparX, preparY, calculate_recallat10 
import os
import scipy.io
#import pickle  # for video env
import pickle5 as pickle # for myPython env
import numpy
from scipy.io import savemat
from matplotlib import pyplot as plt


class Train_AVnet(AVnet):
    
    def __init__(self, model_config , training_config):
        AVnet.__init__(self, model_config)
        
        self.audio_model_name = model_config["audio_model_name"]
        self.visual_model_name = model_config["visual_model_name"]
        self.visual_layer_name = model_config["visual_layer_name"]      
        self.loss = model_config["loss"]
        self.zeropadd = model_config["zeropadd_size"]
        
        self.featuredir = training_config["featuredir"]
        self.featuretype = training_config ["featuretype"]  
        self.outputdir = training_config ["outputdir"]
        self.use_pretrained = training_config ["use_pretrained"]
        self.save_results = training_config["save_results"]
        self.plot_results = training_config["plot_results" ]
        
        self.split = 'testing'
        self.video_name = ''
        self.folder_name = ''
        self.feature_path = '' 
        
        self.dict_errors = {}
        self.recall10_av = 0
        self.recall10_va = 0
        self.trainloss = 1000
        self.valloss = 1000
        self.trainloss_all = []
        self.valloss_all = []
        
        
        
    def initialize_model_outputs(self):
        
        if self.use_pretrained:
            data = scipy.io.loadmat(os.path.join(self.outputdir , 'results.mat'), 
                                    variable_names=['allepochs_valloss','allepochs_trainloss','all_avRecalls', 'all_vaRecalls'])
            allepochs_valloss = data['allepochs_valloss'][0]
            allepochs_trainloss = data['allepochs_trainloss'][0]
            all_avRecalls = data['all_avRecalls'][0]
            all_vaRecalls = data['all_vaRecalls'][0]
            
            self.trainloss_all = numpy.ndarray.tolist(allepochs_trainloss)
            self.valloss_all = numpy.ndarray.tolist(allepochs_valloss)          
            self.av_all = numpy.ndarray.tolist(all_avRecalls)
            self.va_all = numpy.ndarray.tolist(all_vaRecalls)
            
            self.recall_indicator = numpy.max(allepochs_valloss)
            self.val_indicator = numpy.min(allepochs_valloss)
            
        else:
        
            self.trainloss_all = []
            self.valloss_all = []
            self.av_all = []
            self.va_all = []
            self.recall_indicator = 0
            self.val_indicator = 1000
            
        
        
    # def produce_apc_features (self,data):
    #     predictor, apc = self.build_apc(Xshape= (995, 40)) 
    #     predictor.summary()
    #     apc.summary()
    #     predictor.load_weights('%smodel_weights.h5' % self.outputdir)
    #     apc.layers[1].set_weights = predictor.layers[1].get_weights
    #     apc.layers[2].set_weights = predictor.layers[2].get_weights
    #     apc.layers[3].set_weights = predictor.layers[3].get_weights
    #     apc.layers[4].set_weights = predictor.layers[4].get_weights
    #     apc_features = apc.predict(data)
    #     return apc_features
    
    # def train_apc (self ):
    #     l = 5
    #     predictor, apc = self.build_apc(Xshape= (995, 40))         
    #     val_loss_init = 1000
    #     predictor.summary()
       
    #     self.split = "train"
    #     audio_features = self.get_audio_features()
    #     xtrain = audio_features[:,0:-l,:]
    #     ytrain = audio_features[:,l:,:]        
    #     # visual block
    #     #visual_features_train = self.get_visual_features()
    #     self.split = "val"
    #     audio_features = self.get_audio_features()
    #     xval = audio_features[:,0:-l,:]
    #     yval = audio_features[:,l:,:]      
    #     # visual block
    #     #visual_features_val = self.get_visual_features()
    #     for epoch in range(50):
    #         predictor.fit(xtrain, ytrain , epochs=5, shuffle = True, batch_size=120)
    #         val_loss = predictor.evaluate(xval, yval, batch_size=120)       
    #         if val_loss[0] < val_loss_init:
    #             val_loss_init = val_loss[0]
    #             weights = predictor.get_weights()
    #             predictor.set_weights(weights)
    #             predictor.save_weights('%smodel_weights.h5' % self.outputdir)


    def load_dict_onsets (self): 
        input_path = os.path.join(self.featuredir , self.featuretype , self.split)
        input_name =  input_path + '_onsets'  
        with open(input_name, 'rb') as handle:
            self.dict_onsets = pickle.load(handle)
    
        self.error_list = []
        for video_name, value in self.dict_onsets.items():   
            self.video_name = video_name
            self.folder_name = value['folder_name']                                  
            self.feature_path = os.path.join(self.featuredir, self.featuretype , self.split ,  str(self.folder_name))      
            vf = self.load_vf()
            resnet_all = vf['resnet152_avg_pool'] # resnet features for each onset (10*2048)
            for element in resnet_all:
                if len(element) == 0:
                    self.error_list.append(self.video_name)
                    break
                
        for key_to_be_deleted in self.error_list:
            self.dict_onsets.pop(key_to_be_deleted)  


    def load_af (self):       
        af_file = os.path.join(self.feature_path , 'af')   
        with open(af_file, 'rb') as handle:
            af = pickle.load(handle)           
        return af    

    def load_vf (self):       
        vf_file = os.path.join(self.feature_path , 'vf_' + self.visual_model_name)   
        with open(vf_file, 'rb') as handle:
            vf = pickle.load(handle)           
        return vf   
    
    # def update_error_list (self):
    #     self.dict_errors[self.video_name] = self.folder_name
          
    
    def get_audio_features (self):
        
        af_all = [] 
               
        for video_name, value in self.dict_onsets.items():           
            self.video_name = video_name
            self.folder_name = value['folder_name']            
            if len(value['onsets']) == 0:
                self.update_error_list()
            else:
                self.feature_path = os.path.join(self.featuredir , self.featuretype, self.split ,  str(self.folder_name))      
                af = self.load_af()            
                logmel_all = af['logmel40'] 
                logmel = logmel_all#[0:10]
                if self.featuretype == "ann-based":                  
                    af_all.append(logmel)
                elif self.featuretype == "yamnet-based":
                    af_all.extend(logmel)
                    
        if self.featuretype == "ann-based": 
            audio_features = []
            len_of_longest_sequence = 100 * self.zeropadd
            for af_video in af_all:       
                logmel_padded = preparX (af_video,len_of_longest_sequence )
                audio_features.extend(logmel_padded)
            audio_features = numpy.array(audio_features)
        elif self.featuretype == "yamnet-based":
            audio_features = numpy.array(af_all)      
        return audio_features

          
            
    
    def get_visual_features (self):
        vf_all = []
        for video_name, value in self.dict_onsets.items():   
            self.video_name = video_name
            self.folder_name = value['folder_name']                                  
            self.feature_path = os.path.join(self.featuredir, self.featuretype , self.split ,  str(self.folder_name))      
            vf = self.load_vf()
            resnet_all = vf['resnet152_avg_pool'] # resnet features for each onset (10*2048)
            
            if self.featuretype == "ann-based":
                len_of_longest_sequence = self.zeropadd
                resnet_padded = preparY (resnet_all , len_of_longest_sequence) # 50*2048
                vf_all.append(resnet_padded)
            elif self.featuretype == "yamnet-based":
                vf_all.extend(resnet_all) 
            
                
           
                
        if self.featuretype == "ann-based": 
            visual_features = []
            len_of_longest_sequence =  self.zeropadd
            for vf_video in vf_all:       
                resnet_padded = preparY (vf_video , len_of_longest_sequence) # 50*2048
                visual_features.extend(resnet_padded)
            visual_features = numpy.array(visual_features)
        elif self.featuretype == "yamnet-based":
            visual_features = numpy.array(vf_all)      
        
        return visual_features  
    

        
            
    def train(self):       
        self.split = "training"    
        self.load_dict_onsets()
        #APC
        #audio_features_train = self.produce_apc_features (audio_features_train[:,:-5,:])        
        visual_features_train = self.get_visual_features()
        
        audio_features_train = self.get_audio_features()
        Y,X,b = prepare_data (audio_features_train , visual_features_train  , self.loss,  shuffle_data = True)
        del audio_features_train, visual_features_train 
        history =  self.av_model.fit([Y,X], b, shuffle=True, epochs=5, batch_size=128)
        del X,Y
        self.trainloss = history.history['loss'][0]

    def get_input_shapes (self):
        self.split = "testing" 
        self.load_dict_onsets()
        visual_features_test = self.get_visual_features()
        audio_features_test = self.get_audio_features() 
        
        Xshape = numpy.shape(audio_features_test)[1:]        
        Yshape = numpy.shape(visual_features_test)[1:] 
        return [Xshape , Yshape]   
            
    
    def evaluate(self , find_recalls = True):
        
        self.split = "validation"       
        self.load_dict_onsets()
        #APC
        # l = 5
        # audio_features_test = self.produce_apc_features (audio_features_test[:,:-l,:])           
        visual_features_test = self.get_visual_features()
        
        audio_features_test = self.get_audio_features() 
        Ytest, Xtest, b_val = prepare_data (audio_features_test , visual_features_test , self.loss,  shuffle_data = True) 
        del audio_features_test, visual_features_test                  
        self.valloss = self.av_model.evaluate([Ytest,Xtest], b_val, batch_size=128)
        
        
        ########### calculating Recall@10 
        if find_recalls:
            audio_embeddings = self.audio_embedding_model.predict(Xtest)    
            visual_embeddings = self.visual_embedding_model.predict(Ytest) 
            number_of_samples = len(Xtest)
            del Xtest,Ytest
            # audio_embeddings = numpy.squeeze(audio_embeddings)
            # visual_embeddings = numpy.squeeze(visual_embeddings)
            print('checking embedd shape')
            print(audio_embeddings.shape)
            print(visual_embeddings.shape)
                               
            poolsize =  3206
            number_of_trials = 1
            
            recall_av_vec = calculate_recallat10( audio_embeddings, visual_embeddings, number_of_trials,  number_of_samples  , poolsize )          
            recall_va_vec = calculate_recallat10( visual_embeddings , audio_embeddings, number_of_trials,  number_of_samples , poolsize ) 
            self.recall10_av = numpy.mean(recall_av_vec)/(poolsize)
            self.recall10_va = numpy.mean(recall_va_vec)/(poolsize)         
            del audio_embeddings, visual_embeddings
            print('............. results for retrieval ............ av and va ')
            
            print(self.recall10_av)
            print(self.recall10_va)
            
        
        
    def save_model(self):
       
        average_recall = ( self.recall10_av + self.recall10_va ) / 2
        
        if average_recall >= self.recall_indicator: 
            
            self.recall_indicator = average_recall
            self.av_model.save_weights('%smodel_weights.h5' % self.outputdir)
                      
        self.trainloss_all.append(self.trainloss)  
        self.valloss_all.append(self.valloss)
        
        self.av_all.append(self.recall10_av)
        self.va_all.append(self.recall10_va)
        save_file = self.outputdir + 'evaluation_results.mat'
        scipy.io.savemat(save_file, 
                          {'valloss_all':self.valloss_all,'trainloss_all':self.trainloss_all,'av_all':self.av_all,'all_va_all':self.va_all })  
        
        self.make_plot()

        
    def make_plot (self):
        
        plt.figure(figsize = [15,15])
        plot_names = ['training loss','validation loss','av recall@10','va recall@10']
        plot_lists = [self.trainloss_all,self.valloss_all, self.av_all, self.va_all]
        for plot_counter, plot_value in enumerate(plot_lists):
            plt.subplot(2,2,plot_counter+1)
            plt.plot(plot_value)
            plt.title(plot_names[plot_counter])
            plt.xlabel('epoch*5')
            plt.grid()

        plt.savefig(self.outputdir + 'evaluation_plot.pdf', format = 'pdf')            
        
    
    def __call__ (self):
        
        [Xshape , Yshape] = self.get_input_shapes()
        self.visual_embedding_model, self.audio_embedding_model, self.av_model = self.build_network( Xshape , Yshape )
        self.featuretype = 'ann-based'
        
        self.initialize_model_outputs()
        # this must be called for initial evaluation and getting X,Y dimensions
        self.evaluate(find_recalls = True)

        for epoch in range(15):
            print(epoch)
            
            self.train()
            self.evaluate(find_recalls = True)
            if self.save_results:
                self.save_model()
                self.make_plot()


    
 
       
           
  
        

            
            
            
            
            
        
            