import time

import cv2
import numpy as np

import pandas as pd
import torch

from inference.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner
from inference.pose_estimation.lightweight_open_pose.utilities import draw
from spatio_temporal_gcn_learner import SpatioTemporalGCNLearner


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

def pose2numpy(num_current_frames, poses_list):
    C = 2
    T = 300
    V = 18
    M = 1  # num_person_in
    data_numpy = np.zeros((1, C, num_current_frames, V, M))
    skeleton_seq = np.zeros((1, C, T, V, M))
    for t in range(num_current_frames):
        for m in range(len(poses_list[t])):
            data_numpy[0, 0:2, t, :, m] = np.transpose(poses_list[t][m].data)

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

pose_estimator = LightweightOpenPoseLearner()
pose_estimator.download(path=".", verbose=True)
pose_estimator.load("openpose_default")

action_classifier = SpatioTemporalGCNLearner(
    in_channels=2,
    num_point=18,
    graph_type="openpose",
)


model_saved_path = "./temp/stgcn_yagr_checkpoints"
# action_classifier.load(model_saved_path, args.action_checkpoint_name)
action_classifier.load(model_saved_path, "stgcn_yagr-44-945")
      
# path = "/media/lakpa/Storage/youngdusan_data/test_video/videoplayback.mp4"
path = "/media/lakpa/Storage/youngdusan_data/youngdusan_video_data/wind_that_shakes_trees/wind_that_shakes_trees_235.mov"
image_provider = VideoReader(path)  # loading a video or get the camera id 0


f_ind = 0
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

    if counter > 300:
        poses_list.pop(0)
        counter = 300

    if counter > 0:
        skeleton_seq = pose2numpy(counter, poses_list)

        prediction = action_classifier.infer(skeleton_seq)
    
    end_time = time.perf_counter()
    fps = 1.0 / (end_time - start_time)
    avg_fps = 0.8 * fps + 0.2 * fps
    img = cv2.putText(img,"FPS: %.2f" % (avg_fps,),(10, 160),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA,)
    cv2.imshow("Result", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
