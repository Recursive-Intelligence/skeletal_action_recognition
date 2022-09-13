import time

import cv2
import numpy as np


from inference.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner
from inference.pose_estimation.lightweight_open_pose.utilities import draw


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
pose_estimator.download(path=".", verbose=True)
pose_estimator.load("openpose_default")

# path = "/media/lakpa/Storage/youngdusan_data/test_video/videoplayback.mp4"
path = "/media/lakpa/Storage/youngdusan_data/youngdusan_video_data/wind_that_shakes_trees/wind_that_shakes_trees_338.mov"
image_provider = VideoReader(path)  # loading a video or get the camera id 0


f_ind = 0
for img in image_provider:
    start_time = time.perf_counter()
    poses = pose_estimator.infer(img)
    for pose in poses:
        draw(img, pose)
    end_time = time.perf_counter()
    fps = 1.0 / (end_time - start_time)
    avg_fps = 0.8 * fps + 0.2 * fps
    img = cv2.putText(img,"FPS: %.2f" % (avg_fps,),(10, 160),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA,)
    cv2.imshow("Result", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
