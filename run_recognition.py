import time

import cv2
import numpy as np

import pandas as pd
import torch

from inference.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner
from inference.pose_estimation.lightweight_open_pose.utilities import draw
from train_stgcn import SpatioTemporalGCNLearner
from data_extraction import VideoReader, DataExtractor

class RecognitionDemo(object):
    def __init__(self, video_path):

        self.data_extractor = DataExtractor()

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

    def preds2label(self, confidence):
        k = 5
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
        predicted_labels = []
        for img in self.image_provider:
            # img = cv2.resize(img, (1080, 720))
            height, width, _ = img.shape
            # print(height, width)
            start_time = time.perf_counter()
            poses = self.pose_estimator.infer(img)
            
            if not len(poses) == 0:
                for pose in poses:
                    draw(img, pose)
                    
                if len(poses) > 0:
                    counter += 1
                    poses_list.append(poses)

                if counter > self.data_extractor.total_frames:
                    poses_list.pop(0)
                    counter = self.data_extractor.total_frames

                if counter > 0:
                    skeleton_seq = self.data_extractor.pose2numpy(counter, poses_list)

                    prediction = self.action_classifier.infer(skeleton_seq)
                    # print(prediction)
                    skeleton_seq = []
                    category_labels = self.preds2label(prediction.confidence)
                    # print(category_labels)
                    if max(list(category_labels.values())) > 0.7:
                        # self.draw_predgit ss(img, category_labels)

                        predicted_label = torch.argmax(prediction.confidence)
                        predicted_labels.append(predicted_label.item())
                        
            voting_label = ""
            if len(predicted_labels) == 60:
                voting_result = max(predicted_labels, key=predicted_labels.count)
                predicted_labels.clear()
                
                voting_label = self.action_labels[voting_result]
            print(voting_label)
            # if len(voting_label) > 0:
            #     voting_label
            end_time = time.perf_counter()
            fps = 1.0 / (end_time - start_time)
            avg_fps = 0.8 * fps + 0.2 * fps
            img = cv2.putText(img,"FPS: %.2f" % (avg_fps,),(100, 400),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA,)
            img = cv2.putText(img, voting_label,(100, 220),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2)
            cv2.imshow("Result", img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break      

if __name__ == "__main__":
    path = "./resources/test_videos/wind_that_shakes_trees_602.mp4"
    recdem = RecognitionDemo(video_path=path)
    recdem.prediction()