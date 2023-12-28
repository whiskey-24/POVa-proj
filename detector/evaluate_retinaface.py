import sys
import cv2
from glob import glob
import pandas as pd
from pathlib import Path
import pickle
from convert_to_YOLO_OBB import print_progress_bar
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

sys.path.append(f"retinaface/Pytorch_Retinaface")
from retinaface.Pytorch_Retinaface.detect import VehicleDetector, Vehicle

dataset_config = {
    "path_to_dataset": "data/retinaface_eval",
    "crop_bboxes": [[(2320, 1228),
                     (3837, 1655)],
                    [(2708, 359),
                     (3808, 626)],
                    [(22, 1210),
                     (2400, 1567)]],
    "output_path": "out/retina_eval",
    "detector_threshold": 0.5
}


@dataclass
class ImageStats:
    fp: int = 0
    tp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


@dataclass
class Annotation:
    bbox_xyxy: list = field(default_factory=list)
    bbox_xywh: list = field(default_factory=list)
    landmarks: list = field(default_factory=list)


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")


def calculate_metrics(vehicles_in: list[Vehicle],
                      annotations_in: list[Annotation]) -> ImageStats:
    # Convert bounding boxes to a common format (x1, y1, x2, y2)
    vehicle_bboxes = [(vehicle.x1, vehicle.y1, vehicle.x2, vehicle.y2) for vehicle in vehicles_in]
    annotation_bboxes = [(annotation.bbox_xyxy[0], annotation.bbox_xyxy[1], annotation.bbox_xyxy[2], annotation.bbox_xyxy[3]) for annotation in annotations_in]

    # Initialize counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Match detected vehicles with ground truth annotations
    for vehicle_bbox in vehicle_bboxes:
        matched = False
        for annotation_bbox in annotation_bboxes:
            if intersection_over_union(vehicle_bbox, annotation_bbox) > 0.5:
                true_positives += 1
                matched = True
                break
        if not matched:
            false_positives += 1

    false_negatives = len(annotation_bboxes) - true_positives

    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    stats_out = ImageStats(fp=false_positives, tp=true_positives, fn=false_negatives,
                           precision=precision, recall=recall, f1=f1_score)

    return stats_out


# Function to calculate intersection over union (IoU)
def intersection_over_union(box_a: tuple[int, int, int, int],
                            box_b: tuple[int, int, int, int]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou


def graph_outputs(path_to_out_pkl: Path):
    with open(path_to_out_pkl, "rb") as f:
        stats_dict = pickle.load(f)

    df = pd.DataFrame.from_dict(stats_dict, orient="index")
    df = df.sort_index()
    df = df.reset_index()
    df = df.rename(columns={"index": "image_name"})
    # Boxplot of precision, recall, and F1 score
    df.boxplot(column=["precision", "recall", "f1"])
    # Print average precision, recall, and F1 score
    print(df[["precision", "recall", "f1"]].mean())
    plt.show()


if __name__ == "__main__":
    path_to_dataset = Path(dataset_config["path_to_dataset"])
    images_path = path_to_dataset / "images"

    output_path = Path(dataset_config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    path_to_label = path_to_dataset / "label.txt"
    path_to_label_pkl = output_path / "label.pkl"

    mode = "collect"
    if mode == "graph":
        graph_outputs(output_path / "stats.pkl")
        exit(0)
    elif mode == "collect":
        pass
    else:
        raise ValueError("Invalid mode")

    all_imgs = glob(str(images_path / "*.jpg"))
    all_imgs.sort()

    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("test", 640, 480)
    # cv2.setMouseCallback("test", mouse_callback)
    #
    # cv2.imshow("test", cv2.imread(all_imgs[0]))
    # while True:
    #     key = cv2.waitKey(0)
    #     if key == ord('q'):
    #         exit(0)

    # Load from pkl or load from txt annotations
    if path_to_label_pkl.exists():
        with open(path_to_label_pkl, "rb") as f:
            annotations = pickle.load(f)
    else:
        annotations = {}
        with open(path_to_label, "r") as f:
            lines = f.readlines()

        total_lines = len(lines)
        img_name = None
        for idx, line in enumerate(lines):
            print_progress_bar(idx, total_lines, prefix="Processing label.txt")
            if line.startswith("#"):
                img_name = line.split(" ")[1].strip()
                annotations[img_name] = []
            else:
                values = line.strip().split(" ")
                values = [float(value) for value in values]
                x1 = values[0]
                y1 = values[1]
                w = values[2]
                h = values[3]
                # 111 76 43 42 154 103 0.0 140 118 0.0 111 91 0.0 124 76 0.0 132.5 97.0
                # Drop 6th, 9th, 12th and 15th values
                values = values[:6] + values[7:9] + values[10:12] + values[13:15] + values[16:]
                landmarks = values[4:]
                # Divide landmarks into lists of 2
                landmarks = [landmarks[i:i + 2] for i in range(0, len(landmarks), 2)]
                annotation = Annotation()
                annotation.bbox_xyxy = [x1, y1, x1 + w, y1 + h]
                annotation.bbox_xywh = [x1, y1, w, h]
                annotation.landmarks = landmarks
                annotations[img_name].append(annotation)

        print("\nLoaded!")
        with open(path_to_label_pkl, "wb") as f:
            pickle.dump(annotations, f)

    # Load the detector
    detector = VehicleDetector("resnet50", "retinaface/Pytorch_Retinaface/weights/Resnet50_Final.pth",
                               False, dataset_config["detector_threshold"], False)

    output = {}
    total_imgs = len(all_imgs)
    for idx, img_path in enumerate(all_imgs):
        print_progress_bar(idx, total_imgs, prefix="Processing images")
        img_name = img_path.split("/")[-1]

        vehicles = detector.detect_path(img_path)
        annotations_img = annotations[img_name]

        stats = calculate_metrics(vehicles, annotations_img)

        output[img_name] = stats

    print("\nDone, saving to pkl")
    with open(output_path / "stats.pkl", "wb") as f:
        pickle.dump(output, f)


