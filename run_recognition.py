import time

import cv2
import numpy as np

import pandas as pd
import torch

from inference.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner
from inference.pose_estimation.lightweight_open_pose.utilities import draw
from spatio_temporal_gcn_learner import SpatioTemporalGCNLearner
from skeleton_extraction_direct import VideoReader

class RecognitionDemo(object):
    def __init__(self, video_path, channels = 2, total_frames = 300, landmarks = 18, num_persons = 1):
        self.channels = channels
        self.total_frames = total_frames
        self.landmarks = landmarks
        self.num_persons = num_persons

        self.pose_estimator = LightweightOpenPoseLearner()
        self.pose_estimator.download(path=".", verbose=True)
        self.pose_estimator.load("openpose_default")

        self.action_classifier = SpatioTemporalGCNLearner(
            in_channels=2,
            num_point=18,
            graph_type="openpose",
        )
        self.model_saved_path = "./temp/yagr_checkpoints"
        self.action_classifier.load(self.model_saved_path, "yagr-44-495")  

        self.image_provider = VideoReader(video_path)

        self.action_labels = {0 : "big_wind", 1 : "bokbulbok", 2 : "chalseok_chalseok_phaldo", 3 : "chulong_chulong_phaldo", 4 : "crafty_tricks"}

                
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

    def preds2label(self, confidence):
        k = 4
        class_scores, class_inds = torch.topk(confidence, k=k)
        labels = {
            self.action_labels[int(class_inds[j])]: float(class_scores[j].item())
            for j in range(k)
        }
        return labels

    def draw_preds(self, frame, preds):
        for i, (cls, prob) in enumerate(preds.items()):
            cv2.putText(
                frame,
                f"{prob:04.3f} {cls}",
                (10, 40 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

    def prediction(self):
        counter = 0
        poses_list = []
        for img in self.image_provider:
            start_time = time.perf_counter()
            poses = self.pose_estimator.infer(img)
            
            for pose in poses:
                draw(img, pose)
                
            if len(poses) > 0:
                counter += 1
                poses_list.append(poses)

            if counter > self.total_frames:
                poses_list.pop(0)
                counter = self.total_frames

            if counter > 0:
                skeleton_seq = self.pose2numpy(counter, poses_list)

                prediction = self.action_classifier.infer(skeleton_seq)
                category_labels = self.preds2label(prediction.confidence)
                self.draw_preds(img, category_labels)

            predicted_label = torch.argmax(prediction.confidence)
            print(predicted_label)
            
            end_time = time.perf_counter()
            fps = 1.0 / (end_time - start_time)
            avg_fps = 0.8 * fps + 0.2 * fps
            img = cv2.putText(img,"FPS: %.2f" % (avg_fps,),(100, 160),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA,)
            cv2.imshow("Result", img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

if __name__ == "__main__":
    path = "/media/lakpa/Storage/youngdusan_data/all_resized_videos/chulong_chulong_phaldo_361.mp4"

    recdem = RecognitionDemo(video_path=path)
    recdem.prediction()