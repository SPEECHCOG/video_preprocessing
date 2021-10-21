import scipy.io
import numpy
import os

filepath = '/home/hxkhkh/projects/project_5/video/download/missing_videos.mat'
newpath = '/home/hxkhkh/projects/project_5/video/download/train/'

data = scipy.io.loadmat(filepath, variable_names = ['vid_lst', 'vid_url'])
vid_lst_all = data['vid_lst']
vid_url_all = data['vid_url']

for counter in range(len(vid_lst_all)):
    
    vid_newpath =  os.path.join(newpath, (vid_lst_all[counter]) )
    vid_url = vid_url_all [counter]
    os.system(' '.join(("youtube-dl -f bestvideo+worstaudio -o", vid_newpath, vid_url)))
    