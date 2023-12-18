from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import Detection
import cv2


def run_deep_sort_rt(tracker, frame, annotation):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = []
    angle_info = {}

    for index, row in annotation.iterrows():
        bbox_info = row['bbox']
        vehicle_img_coordinate, kernel_size, angle_deg = bbox_info

        
        width, height = kernel_size
        x, y = vehicle_img_coordinate
        bbox = [x, y, width, height]

        confidence = 1.0 
        detection_class = row['Type']

        detections.append((bbox, confidence, detection_class))
        #print(detections)

    print("Number of detections:", len(detections))

    tracks = tracker.update_tracks(detections, frame=frame_rgb)

    print("Number of tracks:", len(tracks))

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue  #skip unconfirmed or lost tracks

        print("Track ID:", track.track_id, "Confirmed:", track.is_confirmed(), "Time since update:", track.time_since_update)
        ltwh = track.to_ltwh()
        print("Drawing bbox:", ltwh)

        #cv2.rectangle: [left, top, right, bottom]
        left, top, width, height  = ltwh[0], ltwh[1], ltwh[2] , ltwh[3] 
        
        right = left + width
        bottom = top + height
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 0), 2)
        cv2.putText(frame, str(track.track_id), (int(left), int(top)), 0, 5e-3 * 200, (255, 0, 255), 2)

    return frame



