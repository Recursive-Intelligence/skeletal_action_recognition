from importlib.resources import path
import time

import cv2
import numpy as np

import pandas as pd
import torch
import natsort
import os
import mediapipe as mp
import pickle
from tqdm import tqdm
import json
class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # self.cap.set(cv2.CAP_PROP_FPS, 20)
        if not self.cap.isOpened():
            raise IOError("Video {} cannot be opened".format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

class DataExtractor(object):
    def __init__(self, channels = 2, total_frames = 300, landmarks = 33, num_persons = 1, videos_path = None, visualize = False, use_skip_frames = False, save_keypoints = False, no_bg = False):  
        self.channels = channels
        self.total_frames = total_frames
        self.landmarks = landmarks
        self.num_persons = num_persons
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.videos_path = videos_path
        self.visualize = visualize
        self.use_skip_frames = use_skip_frames
        self.save_keypoints = save_keypoints
        self.no_bg = no_bg

    def tile(self, a, dim, n_tile):
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
    
    def pose2numpy(self, num_current_frames, landmark_list):
        data_numpy = np.zeros((1, self.channels, num_current_frames, self.landmarks, self.num_persons))
        skeleton_seq = np.zeros((1, self.channels, self.total_frames, self.landmarks, self.num_persons))
        
        for t in range(num_current_frames):
            m = 0 # Only predicted single pose
            # kps = np.array(landmark_list.get(f"frame_{t}"))
            data_numpy[0, 0:2, t, :, m] = np.transpose(np.array(landmark_list.get(f"frame_{t}")))

        # if we have less than num_frames, repeat frames to reach num_frames
        diff = self.total_frames - num_current_frames
        if diff == 0:
            skeleton_seq = data_numpy
        while diff > 0:
            num_tiles = int(diff / num_current_frames)
            if num_tiles > 0:
                data_numpy = self.tile(data_numpy, 2, num_tiles + 1)
                num_current_frames = data_numpy.shape[2]
                diff = self.total_frames - num_current_frames
            elif num_tiles == 0:
                skeleton_seq[:, :, :num_current_frames, :, :] = data_numpy
                for j in range(diff):
                    skeleton_seq[:, :, num_current_frames + j, :, :] = data_numpy[
                        :, :, -1, :, :
                    ]
                break
        return skeleton_seq

    def save_labels(self, sample_names, class_names, out_path, part):
        sample_labels = []
        classnames = sorted(list(class_names.keys()))
        for sample_name in sample_names:
            actioname = ("_").join(sample_name.split("_")[:-1])
            
            for classname in classnames:
                if classname == actioname:
                    new_label_name = sample_name.replace(sample_name, str(class_names[actioname]))
                    sample_labels.append(int(new_label_name))

        with open("{}/{}_label.pkl".format(out_path, part), "wb") as f:
            pickle.dump((sample_names, list(sample_labels)), f)
    
    def skip_n_frames(self, poses_list, required_frame = 60, skip_frame_val = 3):
        
        skipped_frames = []
        for i in range(0, len(poses_list), skip_frame_val):
            skipped_frames.append(poses_list[i])

        extra_frames = required_frame - len(skipped_frames)
        
        if extra_frames < 0:
            remove_index = [i for i in range(1, int(abs(extra_frames) / 2) + 1)]
            for i in remove_index:
                del skipped_frames[i]
                del skipped_frames[-i]

        elif extra_frames > 0:
            copy_index = [i for i in range(1, abs(extra_frames) + 1)]
            for i in copy_index:
                copied_frame = skipped_frames[-i].copy()
                skipped_frames.append(copied_frame)

        return skipped_frames

            
    def extract_data(self, sample_names, total_frames, out_path, part):
        skeleton_data = np.zeros(
            (len(sample_names), 2, total_frames, 33, 1), dtype=np.float32
        )
        for i, s in enumerate(tqdm(sample_names)):
            pose_data = {}
            video_path = os.path.join(self.videos_path, s + ".mp4")

            cap = cv2.VideoCapture(video_path)
            
            counter = 0
            landmarks = []
            frame_keypoints = {}
            # frame_landmarks = []

            with self.mp_pose.Pose(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) as pose:
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        # If loading a video, use 'break' instead of 'continue'.
                        break

                    # Flip the image horizontally for a later selfie-view display, and convert
                    # the BGR image to RGB.
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False
                    results = pose.process(image)
                    if not results.pose_landmarks:
                        continue
                    counter += 1

                    landmark = results.pose_landmarks.landmark
                    landmarks.append(landmark)
                    if counter > total_frames:
                        for cnt in range(counter - total_frames):
                            landmarks.pop(0)
                        counter = total_frames
                        

                cap.release()
                # write data to the json file
                
                for index, landmark in enumerate(landmarks):
                    frame_landmarks = []
                    for keypoint in landmark:
                        frame_landmarks.append([keypoint.x, keypoint.y]) 
                    frame_keypoints[f"frame_{index}"] = frame_landmarks
                
                if counter > 0:
                    # frame_landmarks_dict[count] = frame_landmarks
                    frame_skeleton_seq = self.pose2numpy(counter, frame_keypoints)
                    skeleton_data[i, :, :, :, :] = frame_skeleton_seq
            
        np.save("{}/{}_data_joint.npy".format(out_path, part), skeleton_data)


    def data_gen(self, out_path):

        training_subjects = [i for i in range(1, 301)]

        # class_names = {"big_wind" : 0, "bokbulbok" : 1, "chalseok_chalseok_phaldo" : 2, "chulong_chulong_phaldo" : 3, "crafty_tricks" : 4}
        class_names = {"big_wind" : 0, "bokbulbok" : 1, "chalseok_chalseok_phaldo" : 2, 
                       "chulong_chulong_phaldo" : 3, "crafty_tricks" : 4, "flower_clock" : 5, 
                       "seaweed_in_the_swell_sea" : 6, "sowing_corn_and_driving_pigeons" : 7, 
                       "waves_crashing" : 8, "wind_that_shakes_trees" : 9}

        sample_nums = []
        train_sample_names = []
        val_sample_names = []
        
        files = natsort.natsorted(os.listdir(self.videos_path))
        for file in files:
            sample_name = (file.split("."))[0]
            filenum = int(("_").join((file.split(".")[0]).split("_")[-1]))
            sample_nums.append(filenum)

            istraining = filenum in training_subjects
            if istraining:
                train_sample_names.append(sample_name)
            else:
                val_sample_names.append(sample_name)

        self.extract_data(train_sample_names, self.total_frames, out_path, "train")
        self.extract_data(val_sample_names, self.total_frames, out_path, "val")
        self.save_labels(train_sample_names, class_names, out_path, "train")
        self.save_labels(val_sample_names, class_names, out_path, "val")
        
if __name__ == "__main__":
    videos_path = "/media/lakpa/Storage/youngdusan_data/all_resized_videos"
    # videos_path = "/media/lakpa/Storage/youngdusan_data/test_video"
    # out_path = "./resources/all_classes_60frames_test_videosample"
    out_path = "./resources/mediapipe_data"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    dataextractor = DataExtractor(videos_path=videos_path, visualize=True, save_keypoints = False, no_bg = True, total_frames=300)
    dataextractor.data_gen(out_path)
