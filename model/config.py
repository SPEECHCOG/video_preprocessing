
training_config = {  
  "featuredir": "../../features/youcook2/",
  "featuretype": "yamnet-based",
  "outputdir": "../../model/youcook2/yamnet/eval_ann",
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
    "clip_length" : 10
}


