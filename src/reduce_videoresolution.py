import os

import natsort
from moviepy.editor import *


def reduce_video_resolution(path, destination_dir = ""):
    output_root = ("/").join(path.split("/")[:-1])
    output_path = os.path.join(output_root, destination_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    files = os.listdir(path)
    for file in natsort.natsorted(files):
        filename = file.split(".")[0]
        
        clip = VideoFileClip(os.path.join(path, file))

        final = clip.fx(vfx.resize, width = 1080, height = 720)

        final.write_videofile(f"{output_path}/{filename}.mp4")

if __name__ == "__main__":
    path = "/media/lakpa/Storage/youngdusan_data/youngdusan_all_video_data"
    reduce_video_resolution(path, destination_dir = "./resized_videos")
