

paths = {
  "dataset": "YOUCOOK2",
  "split": "train",
  "datadir": "/tuni/groups/3101050_Specog/corpora/youcook2_dataset",
  "outputdir": "/worktmp2/hxkhkh/current/video/data/youcook2/output",
}


yamnet_settings = {
  "target_sample_rate": 16000,
  "logmel_band": 40,
  "win_length_logmel": 0.025,
  "win_hope_logmel":0.01,
  "win_length_yamnet" : 0.96,
  "win_hope_yamnet" : 0.48,
  "class_names_accepted" : ['Speech', 'Child speech, kid speaking' , 'Conversation', 'Narration, monologue' ],
  "class_index_accepted" : [0,1,2,3],
  "clip_length_seconds" : 10,
  "acceptance_snr" : 0.8
}
