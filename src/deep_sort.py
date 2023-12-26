#https://github.com/nwojke/deep_sort


import cv2
import numpy as np
from deepsort import DeepSortTracker
from deepsort.track import TrackState,Track
from deepsort.detection import Detection
from utils import create_feature_vector
vehicle_types = ['Car', 'Taxi', 'Bus', 'Medium Vehicle', 'Heavy Vehicle', 'Motorcycle']

def run_deep_sort(tracker : DeepSortTracker, frame, annotation):
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
    detections = []
    angle_info = {}

    for index, row in annotation.iterrows():
        bbox_info = row['bbox']
        vehicle_img_coordinate, kernel_size, angle_deg = bbox_info

        
        width, height = kernel_size
        x, y = vehicle_img_coordinate
        bbox = [x, y, width, height]

        confidence = 1.0 #dummy value

        feature = create_feature_vector(row['Type'], row['Angle_img [rad]'])
        print("feature:", feature)

        type_vector = [1 if row['Type'] == vtype else 0 for vtype in vehicle_types]
        print("type_vector:", type_vector)

        detection = Detection(bbox, confidence, type_vector)

        detections.append(detection)

    print("Number of detections:", len(detections))


    tracker.predict()
    tracker.update(detections)
    tracks = tracker.tracks
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        print("Track ID:", track.track_id, "Confirmed:", track.is_confirmed(), "Time since update:", track.time_since_update)
        tlbr = track.to_tlbr()
        print("Drawing bbox:", tlbr)

        top, left, bottom, right = tlbr[0], tlbr[1], tlbr[2] , tlbr[3]

        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 0), 2)
        cv2.putText(frame, str(track.track_id), (int(left), int(top)), 0, 5e-3 * 200, (255, 0, 255), 2)

    return frame