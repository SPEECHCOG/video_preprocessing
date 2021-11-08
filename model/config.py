
training_config = {  
  "featuredir": "../../features/ouput/youcook2/",
  "featuretype": "ann-based",
  "outputdir": "../../model/youcook2/current/",
  "dataset": "YOUCOOK2",
  "use_pretrained": False,
  "save_results" : True,
  "plot_results" : False
}

model_config = { 
    "audio_model_name" : "resDAVEnet",
    "visual_model_name": "resnet152", 
    "visual_layer_name": "avg_pool",
    
    "loss" : "MMS", 
    "zeropadd_size" : 20
}


