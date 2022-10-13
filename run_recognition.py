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
        self.model_saved_path = "./temp/yagr_all_class_60_frames_v2_checkpoints"
        self.action_classifier.load(self.model_saved_path, "yagr_all_class_60_frames_v2-44-945")  

        self.image_provider = VideoReader(video_path)
        self.no_frames = 0
        self.action_labels = {0 : 'big_wind', 1 : 'bokbulbok', 2 : 'chalseok_chalseok_phaldo', 3 : 'chulong_chulong_phaldo', 4 : 'crafty_tricks',
                                5 : 'flower_clock', 6 : 'seaweed_in_the_swell_sea', 7 : 'sowing_corn_and_driving_pigeons', 8 : 'waves_crashing',
                                9 : 'wind_that_shakes_trees'}

    def preds2label(self, confidence):
        k = 10
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
        pred_list = []
        for img in self.image_provider:
            height, width, _ = img.shape
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
                    skeleton_seq = []
                    category_labels = self.preds2label(prediction.confidence)
                    if max(list(category_labels.values())) > 0.85:
                        predicted_label = torch.argmax(prediction.confidence)
                        if counter > 150:
                            pred_text = self.action_labels[predicted_label.item()]
                            pred_list.append(pred_text)
                        else:
                            pred_text = ""   
                        
                        if len(pred_list) > 10:
                            pred_list.clear()
                            img = cv2.putText(img, "",(100, 100),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 0, 255),2)
                        
                        else:                         
                            img = cv2.putText(img, pred_text,(100, 100),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 0, 255),2)                        
            
            end_time = time.perf_counter()
            fps = 1.0 / (end_time - start_time)
            avg_fps = 0.8 * fps + 0.2 * fps
            img = cv2.putText(img,"FPS: %.2f" % (avg_fps,),(10, 60),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA,)
            cv2.imshow("Result", img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break 
                 
    def display_pose_name(self, image, text):
        img = cv2.putText(
            image,
            text,
            (7, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            2,
            (0, 0, 0),
            2,
        )
    
        cv2.imshow("Result", img)
        
    def check_no_frames(self, frames, image, voting_label):
        if frames != 0:
            self.puttext_in_consecutive_frames(frames, image, voting_label)
            
    def puttext_in_consecutive_frames(self, no_frames, image, voting_label):
        if no_frames > 0:
            self.display_pose_name(image, voting_label)
            no_frames -= 1
            self.check_no_frames(no_frames, image, voting_label)
    
    
if __name__ == "__main__":
    path = "./resources/test_videos/wholeaction_v2.mp4"
    recdem = RecognitionDemo(video_path=path)
    recdem.prediction()