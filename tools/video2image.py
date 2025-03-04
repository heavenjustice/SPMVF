import cv2
import os


video_folder_path = './avenue/testing_videos'


for filename in os.listdir(video_folder_path):
    if filename.endswith('.avi'):
        video_path = os.path.join(video_folder_path, filename)
        video = cv2.VideoCapture(video_path)
        success, image = video.read()
        count = 0
        frame_save_path = './frames/' + filename.split('.')[0]
        if not os.path.exists(frame_save_path):
            os.makedirs(frame_save_path)
        while success:
            
            frame_filename = "{:04d}.jpg".format(count)
            frame_path = os.path.join(frame_save_path, frame_filename)
            cv2.imwrite(frame_path, image)
            success, image = video.read()
            count += 1