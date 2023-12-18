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


def load_dataset(dataset_path):
    subfolders = get_dataset_subfolders(dataset_path)

    for subfolder in subfolders:
        frames_folder = os.path.join(subfolder, "Frames")
        annotations_folder = os.path.join(subfolder, "Annotations")
        gt_frames = pair_frames_and_annotations(frames_folder, annotations_folder)

        for frame_file_name, annotation_file_name in gt_frames[190:290]:
            frame_path = os.path.join(frames_folder, frame_file_name)
            annotation_path = os.path.join(annotations_folder, annotation_file_name)

            img = cv2.imread(frame_path)
            df = pd.read_csv(annotation_path)
            
            df['bbox'] = df.apply(lambda row: create_bbox_for_vehicles(row['Type'], (row['x_img [px]'], row['y_img [px]']), np.radians(row['Angle_img [rad]'])), axis=1)

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