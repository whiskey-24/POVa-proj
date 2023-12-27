from glob import glob
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from convert_to_YOLO_OBB import print_progress_bar

dataset_info = {
    "path_to_retinaface": "data/retinaface",
    "path_to_YOLO": "data/yolo_ret1",
}

if __name__ == "__main__":
    path_to_retinaface = Path(dataset_info["path_to_retinaface"])
    path_to_images = path_to_retinaface / "images"

    label_file = path_to_retinaface / "label.txt"
    with open(label_file, "r") as f:
        lines = f.readlines()

    annotations = {}
    img_name = None
    img_width = None
    img_height = None
    total_lines = len(lines)
    for idx, line in enumerate(lines):
        print_progress_bar(idx, total_lines, prefix="Processing label.txt")
        if line.startswith("#"):
            img_name = line.split(" ")[1].strip()
            img_height, img_width = cv2.imread(str(path_to_images / img_name)).shape[:2]
            annotations[img_name] = []
        else:
            values = line.strip().split(" ")
            values = [float(value) for value in values]
            x1 = ((values[0] + (values[0] + values[2])) / 2) / img_width
            y1 = ((values[1] + (values[1] + values[3])) / 2) / img_width
            w = values[2] / img_width
            h = values[3] / img_height
            annotations[img_name].append([x1, y1, w, h])

    print("\nLoaded!")

    yolo_paths = {
        "default": Path(dataset_info["path_to_YOLO"]),
        "train_images": Path(dataset_info["path_to_YOLO"]) / "train" / "images",
        "train_labels": Path(dataset_info["path_to_YOLO"]) / "train" / "labels",
        "test_images": Path(dataset_info["path_to_YOLO"]) / "test" / "images",
        "test_labels": Path(dataset_info["path_to_YOLO"]) / "test" / "labels"
    }

    for value in yolo_paths.values():
        value.mkdir(parents=True, exist_ok=True)

    # Load all images
    all_imgs = glob(str(path_to_images / "*.jpg"))

    # Split the data into training and testing sets
    train_frames, test_frames = train_test_split(all_imgs, test_size=0.2,
                                                 random_state=42)

    total_train_frames = len(train_frames)
    for idx, train_image in enumerate(train_frames):
        print_progress_bar(idx+1, total_train_frames, prefix='Progress train:',
                           suffix='Complete', length=50)
        img_name = train_image.split("/")[-1]
        try:
            img_annot = annotations[img_name]
        except KeyError:
            continue

        # Copy image into train_images
        os.system(f"cp {train_image} {yolo_paths['train_images']}")

        # Create label file
        label_file = yolo_paths["train_labels"] / f"{img_name.split('.')[0]}.txt"
        with open(label_file, "w") as f:
            for annot in img_annot:
                f.write(f"0 {annot[0]} {annot[1]} {annot[2]} {annot[3]}\n")

    print("\nTrain done!")

    total_test_frames = len(test_frames)
    for idx, test_image in enumerate(test_frames):
        print_progress_bar(idx+1, total_test_frames, prefix='Progress test:',
                           suffix='Complete', length=50)

        img_name = test_image.split("/")[-1]
        try:
            img_annot = annotations[img_name]
        except KeyError:
            continue

        # Copy image into train_images
        os.system(f"cp {test_image} {yolo_paths['test_images']}")

        # Create label file
        label_file = yolo_paths["test_labels"] / f"{img_name.split('.')[0]}.txt"
        with open(label_file, "w") as f:
            for annot in img_annot:
                f.write(f"0 {annot[0]} {annot[1]} {annot[2]} {annot[3]}\n")

    print("\nDone!")

