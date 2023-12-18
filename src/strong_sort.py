from strong_sort import StrongSort
import cv2
import numpy as np

def run_strong_sort(tracker, frame, annotation):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = []

    for index, row in annotation.iterrows():
            
            left, top, right, bottom = row['bbox']
            width = right - left
            height = bottom - top  
            bbox = [left, top, width, height]
    
            confidence = 1.0
            detection_class = row['Type'] if 'Type' in row else 'default_class'
    
            detections.append((bbox, confidence, detection_class))
            print(detections)