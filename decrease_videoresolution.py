
# Import everything needed to edit video clips
from moviepy.editor import *
import os 
import natsort

# loading video dsa gfg intro video
# and getting only first 5 seconds
path = "/media/lakpa/Storage/youngdusan_data/youngdusan_all_video_data"
output_root = ("/").join(path.split("/")[:-1])
output_path = os.path.join(output_root, "all_resized_videos")
if not os.path.exists(output_path):
    os.makedirs(output_path)
files = os.listdir(path)
for file in natsort.natsorted(files):
    filename = file.split(".")[0]
    
    clip = VideoFileClip(os.path.join(path, file))

    final = clip.fx(vfx.resize, width = 1080, height = 720)

    final.write_videofile(f"{output_path}/{filename}.mp4")