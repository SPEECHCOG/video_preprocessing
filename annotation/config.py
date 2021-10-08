

paths = {  
  "split": "test",
  "datadir": "/tuni/groups/3101050_Specog/corpora/youcook2_dataset",
  "outputdir": "../../data/youcook2/output/ann-based/",
  "annpath-test": "../../data/youcook2/annotations/youcookii_annotations_test_segments_only.json",
  "annpath-trainval": "../../data/youcook2/annotations/youcookii_annotations_trainval.json"
}

basic = {
    "dataset": "YOUCOOK2",
    "audio_model" : "yamnet", 
    "save_results" : True,
    "plot_results" : False
}

yamnet_settings = {
  "target_sample_rate": 16000,
  "logmel_bands": 40,
  "win_length_logmel": 0.025,
  "win_hope_logmel":0.01,
  "win_length_yamnet" : 0.96,
  "win_hope_yamnet" : 0.48,
  "class_names_accepted" : ['Speech', 'Child speech, kid speaking' , 'Conversation', 'Narration, monologue' ],
  "class_index_accepted" : [0,1,2,3],
  "clip_length_seconds" : 10,
  "acceptance_snr" : 0.8,
  "skip_seconds" : 3
}