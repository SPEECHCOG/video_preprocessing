
training_config = {  
  "featuredir": "../../../features/youcook2/",
  "featuretype": "ann-based",
  "outputdir": "../../../model/youcook2/experiments/",
  "dataset": "YOUCOOK2",
  "use_pretrained": True,
  "save_results" : True,
  "plot_results" : False
}

feature_config = {
    "audio_feature_name": 'embeddings',
    "speech_feature_name": 'logmel40',
    "image_feature_name": 'resnet152_avg_pool',
    }

model_config = { 
    "audio_model_name" : "resDAVEnet",
    "visual_model_name": "resnet152", 
    "visual_layer_name": "avg_pool",
    
    "loss" : "MMS", 
    "clip_length" : 10
}


