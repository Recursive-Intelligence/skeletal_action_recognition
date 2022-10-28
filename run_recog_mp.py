import time

import cv2
import numpy as np

import pandas as pd
import torch

from train_stgcn import SpatioTemporalGCNLearner
import mediapipe as mp

class RecognitionDemo(object):
    def __init__(self, video_path):
        self.action_classifier = SpatioTemporalGCNLearner(
            in_channels=2,
            num_point=18,
            graph_type="openpose",
        )
        self.model_saved_path = "./temp/mediapipe_model_checkpoints"
        self.action_classifier.load(self.model_saved_path, "/mediapipe_model-44-945")  
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
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

    def prediction(self, path):
        counter = 0
        frame_count = 0
        poses_list = []
        pred_list = []
        cap = cv2.VideoCapture(path)
        
        with self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    # If loading a video, use 'break' instead of 'continue'.
                    break
                start_time = time.perf_counter()
                height, width, _ = image.shape

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = pose.process(image)
                if not results.pose_landmarks:
                    continue
                
                
                poses = results.pose_landmarks.landmark
                
                if not len(poses) == 0:
                    for pose in poses:
                        draw(img, pose)
                        
                    if len(poses) > 0:
                        counter += 1
                        frame_count += 1
                        poses_list.append(poses)

                    if counter > self.data_extractor.total_frames:
                        poses_list.pop(0)
                        counter = self.data_extractor.total_frames

                    if counter > 0 and frame_count > 60:
                        skeleton_seq = self.data_extractor.pose2numpy(counter, poses_list)

                        prediction = self.action_classifier.infer(skeleton_seq)
                        skeleton_seq = []
                        category_labels = self.preds2label(prediction.confidence)
                        if max(list(category_labels.values())) > 0.60:
                            predicted_label = torch.argmax(prediction.confidence)
                            if counter > 20:
                                pred_text = self.action_labels[predicted_label.item()]
                                pred_list.append(pred_text)
                            else:
                                pred_text = ""   
                            
                            if len(pred_list) > 40:
                                final_pred = max(pred_list[35:],key=pred_list[35:].count)
                                pred_list.clear()
                                print(final_pred)
                                img = cv2.putText(img, final_pred,(100, 100),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 0, 255),2)
                            
                            else:                         
                                img = cv2.putText(img, "",(100, 100),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 0, 255),2)                        
                
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
    # path = "./videofile.avi"
    recdem = RecognitionDemo()
    recdem.prediction(path = path)