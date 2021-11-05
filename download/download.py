
import os





def download_video ( vid_prefix, vid_name):
  
    vid_url = 'www.youtube.com/watch?v=' + vid_name
    print(f'now is downloading {vid_prefix}')
    os.system(' '.join(("youtube-dl -o", vid_prefix, vid_url)))
  

def set_download_dir (dataset_root,  split, rcp_type, vid_name):
    if not os.path.isdir(os.path.join(dataset_root, split, rcp_type)):
        os.mkdir(os.path.join(dataset_root, split, rcp_type))
    vid_prefix = os.path.join(dataset_root, split, rcp_type, vid_name) 
    return vid_prefix


def get_video_name(line):
    rcp_type,vid_name = line.replace('\n','').split('/')
    return rcp_type,vid_name



