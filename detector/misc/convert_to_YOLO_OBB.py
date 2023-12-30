from glob import glob
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import re

vehicle_dict = {  # type : class_id
    'Motorcycle': 0,  # Motorcycle
    'Car': 1,  # Car
    'Taxi': 1,  # Taxi
    'Bus': 2,  # Bus
    'Medium Vehicle': 3,  # Medium Vehicle
    'Heavy Vehicle': 4,  # Heavy Vehicle
}


def print_progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r', flush=True)
    # Print a new line when the iteration is complete
    if iteration == total:
        print()


if __name__ == "__main__":
    base_dir = Path("data/20181029_D1_0900_0930/20181029_D1_0900_0930")
    base_frames = base_dir / "Frames"
    base_annotations = base_dir / "Annotations"
    yolo_dirs = {"base_dir": base_dir.parent / "yolo",
                 "train_dir": base_dir.parent / "yolo" / "train",
                 "test_dir": base_dir.parent / "yolo" / "test",
                 "train_images_dir": base_dir.parent / "yolo" / "train" / "images",
                 "train_labels_dir": base_dir.parent / "yolo" / "train" / "labels",
                 "test_images_dir": base_dir.parent / "yolo" / "test" / "images",
                 "test_labels_dir": base_dir.parent / "yolo" / "test" / "labels"}

    # # If exists delete the directory
    # if yolo_dirs["base_dir"].exists():
    #     print("Removing the directory {}".format(yolo_dirs["base_dir"]))
    #     os.system(f"rm -rf {yolo_dirs['base_dir']}")

    # Create the directories
    print("Creating the directories")
    for k, v in yolo_dirs.items():
        print(f"Creating the directory {v}")
        v.mkdir(parents=True, exist_ok=True)

    all_frames = glob(str(base_frames / "*.jpg"))

    # Split the data into training and testing sets
    train_frames, test_frames = train_test_split(all_frames, test_size=0.2,
                                                 random_state=42)

    total_train_frames = len(train_frames)
    for idx, train_frame in enumerate(train_frames):
        print_progress_bar(idx+1, total_train_frames, prefix='Progress:',
                           suffix='Complete', length=50)
        # Flush the stdout
        sys.stdout.flush()

        # Skip if the file is already in the directory
        if (yolo_dirs["train_images_dir"] / Path(train_frame).name).exists():
            continue

        os.system(f"cp {train_frame} {yolo_dirs['train_images_dir']}")
        h, w, _ = cv2.imread(train_frame).shape

        df = pd.read_csv(
            str(base_annotations / Path(train_frame).name.replace(".jpg",
                                                                  "_rotated.csv")))
        df["class_id"] = df["Type"].map(vehicle_dict)

        # Drop all rows that have any NaN values
        df.dropna(inplace=True)

        # Extract values from 'p1' to 'p4' columns
        pattern = re.compile(r'\[ *(\d+) *(\d+) *\]')
        df[['p1', 'p2', 'p3', 'p4']] = df[['p1', 'p2', 'p3', 'p4']].map(
            lambda x: re.findall(pattern, x))

        # Create new columns 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'
        df[['x1', 'y1']] = df['p1'].apply(
            lambda x: pd.Series(x[0], dtype=int) if x else pd.Series(
                [None, None], dtype=int))
        df[['x2', 'y2']] = df['p2'].apply(
            lambda x: pd.Series(x[0], dtype=int) if x else pd.Series(
                [None, None], dtype=int))
        df[['x3', 'y3']] = df['p3'].apply(
            lambda x: pd.Series(x[0], dtype=int) if x else pd.Series(
                [None, None], dtype=int))
        df[['x4', 'y4']] = df['p4'].apply(lambda x: pd.Series(x[0], dtype=int))

        # Normalize the values
        df[['x1', 'x2', 'x3', 'x4']] = df[['x1', 'x2', 'x3', 'x4']] / w
        df[['y1', 'y2', 'y3', 'y4']] = df[['y1', 'y2', 'y3', 'y4']] / h

        # Drop all columns except 'class_id', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'
        df = df[['class_id', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']]

        # Save the dataframe as a txt file without header and index and NOT comma separated
        df.to_csv(
            str(yolo_dirs["train_labels_dir"] / Path(train_frame).name.replace(
                ".jpg", ".txt")), header=False, index=False, sep=" ")

    total_test_frames = len(test_frames)
    for idx, test_frame in enumerate(test_frames):
        print_progress_bar(idx+1, total_test_frames, prefix='Progress:',
                           suffix='Complete', length=50)
        # Flush the stdout
        sys.stdout.flush()

        # Skip if the file is already in the directory
        if (yolo_dirs["test_images_dir"] / Path(test_frame).name).exists():
            continue

        os.system(f"cp {test_frame} {yolo_dirs['test_images_dir']}")
        h, w, _ = cv2.imread(test_frame).shape

        df = pd.read_csv(
            str(base_annotations / Path(test_frame).name.replace(".jpg",
                                                                 "_rotated.csv")))
        df["class_id"] = df["Type"].map(vehicle_dict)

        # Drop all rows that have any NaN values
        df.dropna(inplace=True)

        # Extract values from 'p1' to 'p4' columns
        pattern = re.compile(r'\[ *(\d+) *(\d+) *\]')
        df[['p1', 'p2', 'p3', 'p4']] = df[['p1', 'p2', 'p3', 'p4']].map(
            lambda x: re.findall(pattern, x))

        # Create new columns 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'
        df[['x1', 'y1']] = df['p1'].apply(
            lambda x: pd.Series(x[0], dtype=int) if x else pd.Series(
                [None, None], dtype=int))
        df[['x2', 'y2']] = df['p2'].apply(
            lambda x: pd.Series(x[0], dtype=int) if x else pd.Series(
                [None, None], dtype=int))
        df[['x3', 'y3']] = df['p3'].apply(
            lambda x: pd.Series(x[0], dtype=int) if x else pd.Series(
                [None, None], dtype=int))
        df[['x4', 'y4']] = df['p4'].apply(lambda x: pd.Series(x[0], dtype=int))

        # Normalize the values
        df[['x1', 'x2', 'x3', 'x4']] = df[['x1', 'x2', 'x3', 'x4']] / w
        df[['y1', 'y2', 'y3', 'y4']] = df[['y1', 'y2', 'y3', 'y4']] / h

        # Drop all columns except 'class_id', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'
        df = df[['class_id', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']]

        # Save the dataframe as a txt file without header and index and NOT comma separated
        df.to_csv(
            str(yolo_dirs["test_labels_dir"] / Path(test_frame).name.replace(
                ".jpg", ".txt")), header=False, index=False, sep=" ")
