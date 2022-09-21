import time

import cv2
import numpy as np

import pandas as pd
import torch
import natsort
import os
from inference.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner
from inference.pose_estimation.lightweight_open_pose.utilities import draw
import pickle
from tqdm import tqdm

class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError("Video {} cannot be opened".format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

pose_estimator = LightweightOpenPoseLearner()
pose_estimator.load("openpose_default")
# path = "/media/lakpa/Storage/youngdusan_data/test_video/sowingcornforpigeons.mp4"
# image_provider = VideoReader(path)

def tile(a, dim, n_tile):
    a = torch.from_numpy(a)
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    tiled_a = torch.index_select(a, dim, order_index)
    return tiled_a.numpy()

def pose2numpy(num_current_frames, frames, poses_list):
    C = 2
    T = 300
    V = 18
    M = 1  # num_person_in
    data_numpy = np.zeros((1, C, num_current_frames, V, M))
    skeleton_seq = np.zeros((1, C, T, V, M))
    for t in range(num_current_frames):
        # for m in range(len(poses_list[t])):
        m = 0 # Only predicted single pose
        data_numpy[0, 0:2, t, :, m] = np.transpose(poses_list[t][m].data)

    # return data_numpy
    # if we have less than num_frames, repeat frames to reach num_frames
    diff = T - num_current_frames
    if diff == 0:
        skeleton_seq = data_numpy
    while diff > 0:
        num_tiles = int(diff / num_current_frames)
        if num_tiles > 0:
            data_numpy = tile(data_numpy, 2, num_tiles + 1)
            num_current_frames = data_numpy.shape[2]
            diff = T - num_current_frames
        elif num_tiles == 0:
            skeleton_seq[:, :, :num_current_frames, :, :] = data_numpy
            for j in range(diff):
                skeleton_seq[:, :, num_current_frames + j, :, :] = data_numpy[
                    :, :, -1, :, :
                ]
            break
    return skeleton_seq

def save_labels(sample_names, class_names, out_path, part):
    sample_labels = []
    classnames = sorted(list(class_names.keys()))
    for sample_name in sample_names:
        actioname = ("_").join(sample_name.split("_")[:-1])
        
        for classname in classnames:
            if classname == actioname:
                new_label_name = sample_name.replace(sample_name, str(class_names[actioname]))
        
        # for classname in sorted(list(class_names.keys())):
        #     if classname == action_class:
        #         categorical_sample_name = sample_name.replace(sample_name, str(class_names[action_class]))
                sample_labels.append(int(new_label_name))
    # print(set(sample_labels))
    with open("{}/{}_label.pkl".format(out_path, part), "wb") as f:
        pickle.dump((sample_names, list(sample_labels)), f)
        
def extract_data(videos_path, sample_names, total_frames, out_path, part):
    frames = 60
    skeleton_data = np.zeros(
        (len(sample_names), 2, total_frames, 18, 1), dtype=np.float32
    )
    for i, s in enumerate(tqdm(sample_names)):
        video_path = os.path.join(videos_path, s + ".mp4")
        image_provider = VideoReader(video_path)
        
        counter = 0
        poses_list = []        
                
        for img in image_provider:
            start_time = time.perf_counter()
            poses = pose_estimator.infer(img)

            for pose in poses:
                draw(img, pose)

            if len(poses) > 0:
                counter += 1
                poses_list.append(poses)

            cv2.imshow("Result", img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

        if counter > total_frames:
            for cnt in range(counter - total_frames):
                poses_list.pop(0)
            counter = total_frames
        
        if counter > 0:
            
            frame_skeleton_seq = pose2numpy(counter, frames, poses_list)
            skeleton_data[i, :, :, :, :] = frame_skeleton_seq
            
    np.save("{}/{}_data_joint.npy".format(out_path, part), skeleton_data)

def data_gen(videos_path, total_frames, out_path):

    training_subjects = [i for i in range(1, 301)]

    class_names = {"bokbulbok" : 0, "waves_crashing" : 1, "wind_that_shakes_trees" : 2, "sowing_corn_and_driving_pigeons" : 3}

    sample_nums = []
    train_sample_names = []
    val_sample_names = []
    
    files = natsort.natsorted(os.listdir(videos_path))
    for file in files:
        action_class = ("_").join((file.split(".")[0]).split("_")[:-1])
        sample_name = (file.split("."))[0]
        filenum = int(("_").join((file.split(".")[0]).split("_")[-1]))
        sample_nums.append(filenum)

        istraining = filenum in training_subjects
        if istraining:
            train_sample_names.append(sample_name)
        else:
            val_sample_names.append(sample_name)

    extract_data(videos_path, train_sample_names, total_frames, out_path, "train")
    extract_data(videos_path, val_sample_names, total_frames, out_path, "val")
    save_labels(train_sample_names, class_names, out_path, "train")
    save_labels(val_sample_names, class_names, out_path, "val")
    

if __name__ == "__main__":
    videos_path = "/media/lakpa/Storage/youngdusan_data/resized_video"
    total_frames = 300
    out_path = "./resources"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    data_gen(videos_path, total_frames, out_path)