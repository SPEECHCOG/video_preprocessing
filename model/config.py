
paths = {  
  "split": "test",
  "featuredir": "../../data/youcook2/output/",
  "featuretype": "ann-based",
  "outputdir": "../../model/youcook2/"
}

basic = {
    "dataset": "YOUCOOK2",
    "audio_model" : "yamnet",
    "visual_model": "resnet152", 
    "layer_name": "avg_pool",
    "audiochannel" : "resDAVEnet",
    "loss" : "MMS",
    "save_results" : True,
    "plot_results" : False
}

feature_settings = {
  "audio_sample_rate": 16000,
  "zeropadd": 50,
}
