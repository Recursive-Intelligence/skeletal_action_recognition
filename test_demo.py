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

def pose2numpy(num_current_frames, frames, poses_list):
    C = 2
    T = frames
    V = 18
    M = 1  # num_person_in
    data_numpy = np.zeros((1, C, num_current_frames, V, M))
    skeleton_seq = np.zeros((1, C, T, V, M))
    for t in range(num_current_frames):
        # for m in range(len(poses_list[t])):
        m = 0 # Only predicted single pose
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

def prepare_poses(frames, sequence):
    C = 2
    T = frames
    V = 18
    M = 1  # num_person_in
    for index, pose in enumerate(sequence):
        data_numpy = np.zeros((1, C, index + 1, V, M))
    
    for t in range(len(sequence)):
        m = 0
        frame_data = np.transpose(sequence[t][m].data)
        data_numpy[0, 0:2, t, :, m] = frame_data
        
    return data_numpy

def preds2label(confidence):
    k = 3
    class_scores, class_inds = torch.topk(confidence, k=k)
    labels = {
        YGAR_10_CLASSES[int(class_inds[j])]: float(class_scores[j].item())
        for j in range(k)
    }
    return labels

def draw_preds(frame, preds):
    for i, (cls, prob) in enumerate(preds.items()):
        cv2.putText(
            frame,
            f"{prob:04.3f} {cls}",
            (10, 40 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )


pose_estimator = LightweightOpenPoseLearner()
# pose_estimator.download(path=".", verbose=True)
pose_estimator.load("openpose_default")

action_classifier = SpatioTemporalGCNLearner(
    in_channels=2,
    num_point=18,
    graph_type="openpose",
)


model_saved_path = "./temp/stgcn_yagr_checkpoints"
# action_classifier.load(model_saved_path, args.action_checkpoint_name)
action_classifier.load(model_saved_path, "stgcn_yagr-44-180")
      
path = "/media/lakpa/Storage/youngdusan_data/test_video/videoplayback.mp4"
# path = "/media/lakpa/Storage/youngdusan_data/youngdusan_video_data/bokbulbok/bokbulbok_500.mov"
# path = "/media/lakpa/Storage/youngdusan_data/youngdusan_video_data/waves_crashing/waves_crashing_596.mov"
image_provider = VideoReader(path)  # loading a video or get the camera id 0

# YGAR_10_CLASSES = pd.read_csv("datasets/ygar_10classes.csv", verbose=True, index_col=0).to_dict()["name"]
YGAR_10_CLASSES = {0 : "bokbulbok", 1: "waves_crashing"}
def prediction(frames):
    f_ind = 0
    counter = 0
    poses_list = []
    predictions = []
    for img in image_provider:
        start_time = time.perf_counter()
        poses = pose_estimator.infer(img)
        
        for pose in poses:
            draw(img, pose)
            
        if len(poses) > 0:
            counter += 1
            poses_list.append(poses)

            if counter > frames:
                poses_list.pop(0)
                counter = frames
            
            sequence = poses_list[-frames:]
            if len(sequence) == frames:
                skeleton_seq = prepare_poses(frames, sequence)
                # for pose in sequence:
                # skeleton_seq = pose2numpy(counter, frames, pose)
                prediction = action_classifier.infer(skeleton_seq)
                # category_labels = preds2label(prediction.confidence)
                # draw_preds(img, category_labels)
                # print(category_labels)
                predicted_label = torch.argmax(prediction.confidence)
                predictions.append(predicted_label.item())
                
                unique_pred = np.unique(predictions[-30:])[0] 
                
                if unique_pred == predicted_label.item():
                    print(unique_pred)                    
                    predicted_class = YGAR_10_CLASSES[unique_pred]
                    end_time = time.perf_counter()
                    fps = 1.0 / (end_time - start_time)
                    avg_fps = 0.8 * fps + 0.2 * fps
                    img = cv2.putText(img,"FPS: %.2f" % (avg_fps,),(100, 160),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA,)
                    img = cv2.putText(img, predicted_class,(10, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),1)
                    cv2.imshow("Result", img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

prediction(frames = 60)