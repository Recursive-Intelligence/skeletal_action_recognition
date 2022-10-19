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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            raise IOError("Video {} cannot be opened".format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

class DataExtractor(object):
    def __init__(self, channels = 2, total_frames = 300, landmarks = 18, num_persons = 1, videos_path = None, visualize = False, use_skip_frames = False, save_keypoints = False, no_bg = False):  
        self.channels = channels
        self.total_frames = total_frames
        self.landmarks = landmarks
        self.num_persons = num_persons
        self.pose_estimator = LightweightOpenPoseLearner()
        self.pose_estimator.load("openpose_default")
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
    
    def pose2numpy(self, num_current_frames, poses_list):
        data_numpy = np.zeros((1, self.channels, num_current_frames, self.landmarks, self.num_persons))
        skeleton_seq = np.zeros((1, self.channels, self.total_frames, self.landmarks, self.num_persons))
        
        for t in range(num_current_frames):
            m = 0 # Only predicted single pose
            data_numpy[0, 0:2, t, :, m] = np.transpose(poses_list[t][m].data)

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
            (len(sample_names), 2, total_frames, 18, 1), dtype=np.float32
        )
        bg_img = cv2.resize(cv2.imread("./resources/bg_image.jpg"), (1080, 720))
        for i, s in enumerate(tqdm(sample_names)):
            video_path = os.path.join(self.videos_path, s + ".mp4")
            image_provider = VideoReader(video_path)
            
            counter = 0
            poses_list = []        
            for img in image_provider:
                start_time = time.perf_counter()
                poses = self.pose_estimator.infer(img)
                
                if self.no_bg:
                    img = bg_img.copy()
                    
                for pose in poses:
                        draw(img, pose)

                if len(poses) > 0:
                    counter += 1
                    poses_list.append(poses)
                
                if self.visualize:
                    cv2.imshow("Result", img)
                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        break
            
            if self.use_skip_frames:
                poses_list = self.skip_n_frames(poses_list=poses_list)
                counter = len(poses_list)
            else:
                if counter > total_frames:
                    for cnt in range(counter - total_frames):
                        poses_list.pop(0)
                    counter = total_frames
            
            if self.save_keypoints:
                json_output_path = "./resources/json_keypoints_data"
                if not os.path.exists(json_output_path):
                    os.makedirs(json_output_path)
                video_keypoints = {}
                for index, value in enumerate(poses_list):
                    video_keypoints[f"frame_{index}"] = (poses_list[index][0].data).tolist()
                with open(f"{json_output_path}/{s}.json", "w") as f:
                    json.dump(video_keypoints, f, indent=3)
                
            if counter > 0:
                frame_skeleton_seq = self.pose2numpy(counter, poses_list)
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
    # videos_path = "/media/lakpa/Storage/youngdusan_data/all_resized_videos"
    videos_path = "/media/lakpa/Storage/youngdusan_data/test_video"
    out_path = "./resources/all_classes_60frames_test_videosample"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    dataextractor = DataExtractor(videos_path=videos_path, visualize=True, save_keypoints = False, no_bg = True, total_frames=60)
    dataextractor.data_gen(out_path)