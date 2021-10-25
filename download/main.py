"""
This file searches for missing videos and downloads them in separate folder

"""
split = 'train'
import os

videolist_path = "../../data/vgs_videolists/youcook2/splits/"
vid_file = os.path.join(videolist_path, split + '_list.txt' )



    
    
dataset_root = "/run/media/hxkhkh/b756dee3-de7e-4cdd-883b-ac95a8d00407/vgs/data/youcook2_dataset"
newpath = "/worktmp/khorrami/project_5/video/data/youcook2/splits/download/"



# downloaded_videos_files = os.path.join(dataset_root, split)
# downloaded_video_names = os.listdir(downloaded_videos_files)

# downloaded_video_purenames = []
# for fullname in downloaded_video_names:
#     name1 = fullname[4:]
#     name2 = name1.split('.')
#     downloaded_video_purenames.append(name2[0])


video_names = []
missing_vid_lst = []

list1 = []
list2 = []

with open(vid_file) as f:
    lines = f.readlines()
    for line in lines:
        rcp_type,vid_name = line.replace('\n','').split('/')
        print(vid_name)
        video_names.append(vid_name)
        
        
        vid_url = 'www.youtube.com/watch?v='+vid_name
        vid_prefix = os.path.join(dataset_root, split, rcp_type + '_' + vid_name) 
        vid_newpath = os.path.join(newpath, split, rcp_type + '_' + vid_name) 
        #os.system(' '.join(("youtube-dl -o", vid_prefix, vid_url)))

            # check if the video is downloaded
        if os.path.exists(vid_prefix + '.mp4') or os.path.exists(vid_prefix + '.mkv') or os.path.exists(vid_prefix + '.webm'):
            print('[INFO] Downloaded {} video {}'.format(split, vid_name))
        else:
            missing_vid_lst.append(rcp_type + '_' + vid_name)
            list1.append(rcp_type + '_' + vid_name)
            list2.append(vid_url)
            os.system(' '.join(("youtube-dl -o", vid_newpath, vid_url)))
            print('[INFO] Cannot download {} video {}'.format(split, vid_name))
import scipy.io

scipy.io.savemat('/worktmp/khorrami/project_5/video/data/youcook2/missing_videos.mat', {'vid_lst':list1,'vid_url':list2})    
# count = 0
# missing_videos = []
# for item in video_names:
#     correct_item = item.strip()
#     if correct_item not in downloaded_video_purenames:
#         count += 1
#         print(correct_item)
#         missing_videos.append(correct_item)

