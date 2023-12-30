import os
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
'''
FRAME_WIDTH = 4096
FRAME_HEIGHT = 2100

VEHICLE_SIZES = {
    'Taxi': (0.01, 0.02),  
    'Motorcycle': (0.0075, 0.015),
    'Heavy Vehicle': (0.02, 0.04),
    'Car': (0.01, 0.02),
    'Bus': (0.02, 0.04),
    'Medium Vehicle': (0.015, 0.03),
}

def load_dataset_all(dataset_path):
    frames = []
    all_annotations = []  # This will hold all annotations DataFrames

    subfolders = get_dataset_subfolders(dataset_path)

    for subfolder in subfolders:

        frames_folder = os.path.join(subfolder, "Frames")
        annotations_folder = os.path.join(subfolder, "Annotations")
        gt_frames = pair_frames_and_annotations(frames_folder, annotations_folder)

        for frame_file_name, annotation_file_name in gt_frames:
            img = cv2.imread(os.path.join(frames_folder, frame_file_name))
            if img is not None:
                frames.append(img)
                annotation_path = os.path.join(annotations_folder, annotation_file_name)
                df = pd.read_csv(annotation_path)
                df['x_norm'] = df['x_img [px]'] / FRAME_WIDTH
                df['y_norm'] = df['y_img [px]'] / FRAME_HEIGHT
                df['bbox'] = df.apply(lambda row: estimate_bbox(row['x_norm'], row['y_norm'], row['Type']), axis=1)
                all_annotations.append(df) 

    return frames, all_annotations



def load_dataset(dataset_path):
    subfolders = get_dataset_subfolders(dataset_path)

    for subfolder in subfolders:
        frames_folder = os.path.join(subfolder, "Frames")
        annotations_folder = os.path.join(subfolder, "Annotations")
        gt_frames = pair_frames_and_annotations(frames_folder, annotations_folder)

        for frame_file_name, annotation_file_name in gt_frames[:200]:
            frame_path = os.path.join(frames_folder, frame_file_name)
            annotation_path = os.path.join(annotations_folder, annotation_file_name)

            img = cv2.imread(frame_path)
            df = pd.read_csv(annotation_path)
            df['x_norm'] = df['x_img [px]'] / FRAME_WIDTH
            df['y_norm'] = df['y_img [px]'] / FRAME_HEIGHT
            df['bbox'] = df.apply(lambda row: estimate_bbox(row['x_norm'], row['y_norm'], row['Type']), axis=1)

            yield img, df


def estimate_bbox(x_norm, y_norm, object_type):
    norm_width, norm_height = VEHICLE_SIZES.get(object_type, (0.01, 0.02))
    perspective_scale = 1 - (y_norm * 0.5)  
    width_px = (norm_width * FRAME_WIDTH) * perspective_scale
    height_px = (norm_height * FRAME_HEIGHT) * perspective_scale
    x_min = int((x_norm * FRAME_WIDTH) - (width_px / 2))
    y_min = int((y_norm * FRAME_HEIGHT) - (height_px / 2))
    x_max = int((x_norm * FRAME_WIDTH) + (width_px / 2))
    y_max = int((y_norm * FRAME_HEIGHT) + (height_px / 2))
    return [x_min, y_min, x_max, y_max]
'''

vehicle_bbox_info = { 
    'Motorcycle': [0, (18, 10), 0],    
    'Car': [1, (40, 20), 3],    
    'Taxi': [2, (40, 20), 3],    
    'Bus': [3, (110, 30), 50],  
    'Medium Vehicle': [4, (50, 20), 8],    
    'Heavy Vehicle': [5, (50, 25), 10],   
}

def create_bbox_for_vehicles(vehicle_type, vehicle_img_coordinate, vehicle_direction_angle):
    '''
        vehicle_type: The type of the vehicle either in string format of in integer
        ex) Motorcycle = 0
            Car = 1
            Taxi =2
            Bus = 3
            Medium Vehicle = 4
            Heavy Vehicle = 5
    '''

    vehicle_class = vehicle_bbox_info[vehicle_type][0]
    kernel_size = vehicle_bbox_info[vehicle_type][1]
    l = vehicle_bbox_info[vehicle_type][2]

    new_x = vehicle_img_coordinate[0] - l * np.cos(vehicle_direction_angle)
    new_y = vehicle_img_coordinate[1] - l * np.sin(vehicle_direction_angle)
    vehicle_img_coordinate = (new_x, new_y)

    #cv2.boxPoints((centerX, centerY), (w, h), angle_in_degree)
    box = (vehicle_img_coordinate, kernel_size, np.rad2deg(vehicle_direction_angle))

    return box

def create_ltwh_for_vehicles(vehicle_type, vehicle_img_coordinate):

    kernel_size = vehicle_bbox_info[vehicle_type][1]

    # Calculate the half-width and half-height
    half_width = kernel_size[0] / 2
    half_height = kernel_size[1] / 2

    # Calculate left, top, width, and height for the bounding box
    left = vehicle_img_coordinate[0] - half_width
    top = vehicle_img_coordinate[1] - half_height
    width = kernel_size[0]
    height = kernel_size[1]

    # Create the bounding box in LTWH format
    bbox = (left, top, width, height)

    return bbox

def load_dataset(dataset_path):
    subfolders = get_dataset_subfolders(dataset_path)

    for subfolder in subfolders[:-1]:
        frames_folder = os.path.join(subfolder, "Frames")
        annotations_folder = os.path.join(subfolder, "Annotations")
        gt_frames = pair_frames_and_annotations(frames_folder, annotations_folder)

        for frame_file_name, annotation_file_name in gt_frames[:50]: 
            frame_path = os.path.join(frames_folder, frame_file_name)
            annotation_path = os.path.join(annotations_folder, annotation_file_name)

            img = cv2.imread(frame_path)
            df = pd.read_csv(annotation_path)
            
            df['bbox'] = df.apply(lambda row: create_bbox_for_vehicles(row['Type'], (row['x_img [px]'], row['y_img [px]']), np.radians(row['Angle_img [rad]'])), axis=1)
            df['ltwh'] = df.apply(lambda row: create_ltwh_for_vehicles(row['Type'], (row['x_img [px]'], row['y_img [px]'])), axis=1)
            yield img, df


def get_sorted_filenames(directory):

    return sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

def pair_frames_and_annotations(frames_directory, annotations_directory):

    frame_files = get_sorted_filenames(frames_directory)
    annotation_files = get_sorted_filenames(annotations_directory)

    paired_files = []
    for frame_file in frame_files:
        annotation_file = frame_file.replace('.jpg', '.csv')  

        if annotation_file in annotation_files:
            paired_files.append((frame_file, annotation_file))

    return paired_files

def get_dataset_subfolders(dataset_path):
   
    subfolders = []
    for entry in os.listdir(dataset_path):
        full_path = os.path.join(dataset_path, entry)  
        if os.path.isdir(full_path):
            subfolders.append(full_path)

    return subfolders

def create_video(frames, output_file='output.mp4', fps=20.0):

    if not frames:
        raise ValueError("The frames list is empty.")

    height, width, layers = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for frame in frames:
        print("Writing frame...")
        out.write(frame)

    out.release()

    print(f"Video saved as {output_file}")

def create_feature_vector(vehicle_type, angle_rad):
    vehicle_types = ['Car', 'Taxi', 'Bus', 'Medium Vehicle', 'Heavy Vehicle', 'Motorcycle']
    type_vector = [1 if vehicle_type == vtype else 0 for vtype in vehicle_types]

    # Normalize the angle
    normalized_angle = angle_rad / (2 * np.pi)

    # Combine into a single feature vector
    feature_vector = type_vector + [normalized_angle]

    return feature_vector


def crop_frame_and_filter_vehicles(img, df, top, left, bottom, right):
    """
    Crops the frame and filters out vehicles not within the cropped area.

    :param img: The original image frame.
    :param df: DataFrame containing vehicle information and bounding boxes.
    :param top: The top y-coordinate for cropping.
    :param left: The left x-coordinate for cropping.
    :param bottom: The bottom y-coordinate for cropping.
    :param right: The right x-coordinate for cropping.
    :return: Cropped image and filtered DataFrame.
    """
    # Crop the frame
    cropped_img = img[top:bottom, left:right]

    # Define a function to check if a vehicle is within the cropped area
    def is_vehicle_within_cropped_area(row):
        bbox = row['bbox']
        bbox_center = bbox[0]
        bbox_size = bbox[1]

        # Calculate bbox corners
        top_left_x = bbox_center[0] - bbox_size[0] / 2
        top_left_y = bbox_center[1] - bbox_size[1] / 2
        bottom_right_x = bbox_center[0] + bbox_size[0] / 2
        bottom_right_y = bbox_center[1] + bbox_size[1] / 2

        # Check if the vehicle is within the cropped area
        return (left <= top_left_x <= right and top <= top_left_y <= bottom) or \
               (left <= bottom_right_x <= right and top <= bottom_right_y <= bottom)

    # Filter the DataFrame
    filtered_df = df[df.apply(is_vehicle_within_cropped_area, axis=1)]

    return cropped_img, filtered_df


def evaluate_tracks(vehicle_tracks, vehicle_detections_gt, iou_threshold):
    total_matched_bboxes = 0
    iou_sum = 0

    # Collect all time keys
    time_keys = set()
    for track in vehicle_tracks.values():
        time_keys.update(track.trajectory.keys())
    for detection in vehicle_detections_gt.values():
        time_keys.update(detection.list_of_bboxes.keys())

    # Iterate over each time frame
    for time in time_keys:  
        matched_pairs = []  # To store matched track-detection pairs for this time frame

        for track_id, track in vehicle_tracks.items():
            if time in track.trajectory:
                track_bbox = track.trajectory[time]
                best_iou = 0
                best_gt_bbox = None

                for detection in vehicle_detections_gt.values():
                    if time in detection.list_of_bboxes:
                        gt_bbox = detection.list_of_bboxes[time]
                        iou = get_iou(track_bbox, gt_bbox)

                        if iou > best_iou:
                            best_iou = iou
                            best_gt_bbox = gt_bbox

                if best_iou > iou_threshold:
                    matched_pairs.append((track_bbox, best_gt_bbox))
                    iou_sum += best_iou

        total_matched_bboxes += len(matched_pairs)

    # Calculate average IoU across all matched pairs
    average_iou = iou_sum / total_matched_bboxes if total_matched_bboxes > 0 else 0
    return average_iou




def get_iou(bbox1, bbox2):
    # Assuming bbox format is (x, y, width, height)
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0
    return iou
