

import json
import os
import numpy

split = 'validation'


#........................................................... reading json files

# datadir = "/run/media/hxkhkh/b756dee3-de7e-4cdd-883b-ac95a8d00407/video/youcook2"
# test_ann_file = "/run/media/hxkhkh/b756dee3-de7e-4cdd-883b-ac95a8d00407/video/youcook2/annotations/youcookii_annotations_test_segments_only.json"
# train_ann_file = "/run/media/hxkhkh/b756dee3-de7e-4cdd-883b-ac95a8d00407/video/youcook2/annotations/youcookii_annotations_trainval.json"


datadir = "../../data/youcook2"
test_ann_file = "../../data/youcook2/annotations/youcookii_annotations_test_segments_only.json"
train_ann_file = "../../data/youcook2/annotations/youcookii_annotations_trainval.json"


with open(train_ann_file) as handle:
    annotations_trainval = json.load(handle) 

with open(test_ann_file) as handle:
    annotations_test = json.load(handle) 

if split == 'training' or split== 'validation':
    database_anns = annotations_trainval['database']
else:
    database_anns = annotations_test['database']
    
#.......................................................... listing video names
    
def create_video_list (datadir, split ):        
    video_dir = os.path.join(datadir, 'videos' , split) 
    video_recepies = os.listdir(video_dir)
    video_list = []
    for rec in video_recepies:
        files = os.listdir(os.path.join(video_dir, rec))
        video_list.extend([os.path.join(split , rec ,f) for f in files])
    return video_list
    
def create_video_names(video_list):
    video_names = []
    for item in video_list:
        item_splitted = item.split('/')
        video_name_extended = item_splitted[-1]
        video_name = video_name_extended.split('.')[0]
        video_names.append(video_name)
    return video_names    
    
#.......................................................... Analysis


video_list = create_video_list (datadir, split )
video_names = create_video_names (video_list)

all_durations = []
all_id_counts = []
all_sentences = []
all_clip_durations = 0
counter_video = 0
all_info = {}
for video_fullname in video_list:
    video_name = video_fullname.split('/')[-1]
    vid_name = video_name.split('.')[0]
    info = database_anns[vid_name] 
    duration = info['duration']
    anns = info['annotations']
    id_counts = len(anns)
    video_sentences = []
    for clip in anns:
        sentence = clip['sentence']
        video_sentences.append(sentence)
        
        clip_time = clip['segment']
        clip_duration = clip_time[1]-clip_time[0]
        all_clip_durations += clip_duration
    
    all_durations.append(duration)
    all_id_counts.append(id_counts)
    all_sentences.append(video_sentences)
    video_info = {}
    video_info['video_full_name'] = video_fullname
    video_info['video_name'] = vid_name
    video_info['id_counts'] = id_counts
    video_info['sentences'] = video_sentences
    all_info [counter_video] = video_info
    counter_video +=1

# extracting averages

number_of_all_clips =  sum(all_id_counts)   
duration_average = round(numpy.mean(all_durations),2)
duration_std = round(numpy.std(all_durations),2)

segments_average = round(numpy.mean(all_id_counts),2)
segments_std = round(numpy.std(all_id_counts),2)

duration_clip_average = round( (all_clip_durations / number_of_all_clips) ,2)
print(f'\n........ In {split} data ')
print(f'\n There are in total " {number_of_all_clips} "  clips (sentences) ')
print(f'\n There are in total " {all_clip_durations} " seconds (images) ')
print(f'\n There are in average " {duration_clip_average} " images per sentence ')
print(f'\n Average video duration is  {duration_average} +(-) {duration_std} seconds ')
print(f'\n Average number of clips per video is  {segments_average} +(-) {segments_std} ')

