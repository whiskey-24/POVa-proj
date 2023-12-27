from glob import glob
from pathlib import Path
from convert_to_YOLO_OBB import print_progress_bar

vehicle_dict = {  # class_id : type
    0: 'Motorcycle',  # Motorcycle
    1: 'Car',  # Car
    2: 'Bus',  # Bus
    3: 'Medium Vehicle',  # Medium Vehicle
    4: 'Heavy Vehicle',  # Heavy Vehicle
}

if __name__ == "__main__":
    path_to_old_labels = "data/20181029_D1_0900_0930/yolo/train/labels_old"
    path_to_new_labels = Path(path_to_old_labels).parent / "labels"
    path_to_new_labels.mkdir(parents=True, exist_ok=True)

    all_labels = glob(path_to_old_labels + "/*.txt")

    total_labels = len(all_labels)

    for idx, label_path in enumerate(all_labels):
        print_progress_bar(idx, total_labels, prefix='Progress:', suffix='Complete', length=50)

        with open(label_path, "r") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            line = line.strip().split(" ")
            points = line[1:9]
            points.append(vehicle_dict[int(line[0])])
            points.append("0")
            new_lines.append(" ".join(points) + "\n")
        with open(str(path_to_new_labels) + "/" + Path(label_path).name, "w") as f:
            f.writelines(new_lines)


