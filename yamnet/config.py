#  "../../data/input/"

paths = {  
  "split": "test",
  "datadir": "/tuni/groups/3101050_Specog/corpora/youcook2_dataset",
  "outputdir": "../../data/youcook2/output/yamnet-based/",
  "exp_name" : "exp3"
}

basic = {
    "dataset": "YOUCOOK2",
    "audio_model" : "yamnet",
    "run_speech_detection" : False,
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
  "clip_length_seconds" : 15,
  "acceptance_snr" : 0.7,
  "skip_seconds" : 3,
  "accepted_overlap_second": 0
}
