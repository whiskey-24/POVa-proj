#https://github.com/levan92/deep_sort_realtime/tree/master

from deep_sort_realtime.deepsort_tracker import DeepSort
from tracker.src.utils_tracker import create_feature_vector
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import Detection
from tracker.src.vehicle import Vehicle_Detection_gt, Vehicle_Track


def run_deep_sort_rt(tracker, frame, annotation, vehicle_tracks, vehicle_detections_gt):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = []
    
    for index, row in annotation.iterrows():
        '''bbox_info = row['bbox']
        vehicle_img_coordinate, kernel_size, angle_deg = bbox_info

        width, height = kernel_size
        x, y = vehicle_img_coordinate
        bbox = [x, y, width, height]
        '''
        bbox = row['ltwh']

        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 255), 3)
        cv2.putText(frame, str(row['ID']), (int(bbox[0] + bbox[2]), int(bbox[1])), 0, 5e-3 * 200, (0, 0, 0), 2)

        confidence = 1.0 
        detection_class = row['Type']
        feature = create_feature_vector(row['Type'], row['Angle_img [rad]'])

        detections.append((bbox, confidence, feature))

        if row['ID'] not in vehicle_detections_gt:
            vehicle_detections_gt[row['ID']] = Vehicle_Detection_gt(row['ID'], row['Type'])

        vehicle_detections_gt[row['ID']].add_position_at_time(bbox, row['Time [s]'])

    tracks = tracker.update_tracks(detections, frame=frame_rgb)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1: 
            continue  #skip unconfirmed or lost tracks
           
        ltwh = track.to_ltwh()

        left, top, width, height  = ltwh[0], ltwh[1], ltwh[2] , ltwh[3] 
        right = left + width
        bottom = top + height

        if track.track_id not in vehicle_tracks:
            vehicle_tracks[track.track_id] = Vehicle_Track(track.track_id)

        vehicle_tracks[track.track_id].add_position_at_time(ltwh, row['Time [s]'])
        vehicle_tracks[track.track_id].decode_type(ltwh)
        vehicle_tracks[track.track_id].add_original_ltwh(track.original_ltwh)

        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 0), 2)
        cv2.putText(frame, str(track.track_id), (int(left), int(top)), 0, 5e-3 * 200, (255, 0, 255), 2)

    return frame



