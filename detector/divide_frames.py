import cv2
from glob import glob
from pathlib import Path
import pandas as pd
import re
from convert_to_YOLO_OBB import print_progress_bar


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")


dataset_config = {
    "path_to_dataset": "/home/whiskey/Documents/2Mit/POVa/POVa-proj/detector/data/20181029_D1_0900_0930/20181029_D1_0900_0930",
    "crop_bboxes": [
        [(9, 879),
         (1813, 1267)],
        [(1923, 935),
         (3752, 1220)],
        [(1751, 1167),
         (1985, 2074)],
        [(1757, 15),
         (1970, 985)]],
    "output_path": "/home/whiskey/Documents/2Mit/POVa/POVa-proj/detector/data/retinaface/images",
    "dateset_id": 1}

# dataset_config = {
#     "path_to_dataset": "/home/whiskey/Documents/2Mit/POVa/POVa-proj/detector/data/20181029_D10_0900_0930",
#     "crop_bboxes": [[(2154, 1385),
#                      (3200, 1709)],
#                     [(2450, 692),
#                      (3190, 1269)],
#                     [(3306, 1014),
#                      (4080, 1450)]
#                     ],
#     "output_path": "/home/whiskey/Documents/2Mit/POVa/POVa-proj/detector/data/retinaface/images",
#     "dateset_id": 10}


def sanity_check(path_to_labels: str):
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("test", 640, 480)
    cv2.setMouseCallback("test", mouse_callback)

    path_to_imgs = Path(path_to_labels).parent / "images"
    all_annotations: dict[str, list[list[float]]] = {}
    with open(path_to_labels, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if line.startswith("#"):
                img_name = str(path_to_imgs / line[2:])
                all_annotations[img_name] = []
            else:
                line = line.split(" ")
                line = [float(x) for x in line]
                all_annotations[img_name].append(line)

    for img_name, annotations in all_annotations.items():
        img = cv2.imread(img_name)
        # Annotations are in the format: x1 y1 w h x1 y1 0.0 x2 y2 0.0 x3 y3 0.0 x4 y4 0.0 x_center y_center
        lndmrk_clrs = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                       (0, 255, 255), (255, 255, 255)]
        for annotation in annotations:
            cv2.rectangle(img, (int(annotation[0]), int(annotation[1])),
                            (int(annotation[0] + annotation[2]), int(annotation[1] + annotation[3])),
                            (0, 0, 255), 2)
            cv2.line(img, (int(annotation[4]), int(annotation[5])),
                        (int(annotation[7]), int(annotation[8])),
                        (0, 255, 0), 2)
            cv2.line(img, (int(annotation[7]), int(annotation[8])),
                     (int(annotation[10]), int(annotation[11])),
                     (0, 255, 0), 2)
            cv2.line(img, (int(annotation[10]), int(annotation[11])),
                     (int(annotation[13]), int(annotation[14])),
                     (0, 255, 0), 2)
            cv2.line(img, (int(annotation[13]), int(annotation[14])),
                     (int(annotation[4]), int(annotation[5])),
                     (0, 255, 0), 2)
            cv2.circle(img, (int(annotation[4]), int(annotation[5])), 2, lndmrk_clrs[0], 2)
            cv2.circle(img, (int(annotation[7]), int(annotation[8])), 2, lndmrk_clrs[1], 2)
            cv2.circle(img, (int(annotation[10]), int(annotation[11])), 2, lndmrk_clrs[2], 2)
            cv2.circle(img, (int(annotation[13]), int(annotation[14])), 2, lndmrk_clrs[3], 2)
            cv2.circle(img, (int(annotation[16]), int(annotation[17])), 2, lndmrk_clrs[4], 2)
        cv2.imshow("test", img)
        while True:
            key = cv2.waitKey(0)
            if key == ord("q"):
                exit(0)
            elif key == ord("n"):
                break


if __name__ == "__main__":
    # sanity_check("data/retinaface/label.txt")
    path_to_dataset = dataset_config["path_to_dataset"]
    path_to_frames = Path(path_to_dataset) / "Frames"
    path_to_annot = Path(path_to_dataset) / "Annotations"

    all_frames = glob(str(path_to_frames / "*.jpg"))
    # Remove all paths that have underscore in the filename
    all_frames = [x for x in all_frames if "_" not in Path(x).name]
    all_frames.sort()

    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("test", 640, 480)
    # cv2.setMouseCallback("test", mouse_callback)
    # cv2.imshow("test", cv2.imread(all_frames[0]))
    # while True:
    #     key = cv2.waitKey(0)
    #     if key == ord("q"):
    #         exit(0)

    crop_bboxes = dataset_config["crop_bboxes"]

    total_frames = len(all_frames)

    # Format of out_lines:
    # For each file there is a # followed by the path to the file from the path_to_dataset, e.g.: # 00001.jpg
    # Followed by one line for each bounding box in the file, coordinates of their corners and middle point
    out_lines = []
    for idx, frame in enumerate(all_frames):
        print_progress_bar(idx, total_frames, prefix="Progress:",
                           suffix="Complete", length=50)
        image = cv2.imread(frame)
        frame_name = Path(frame).name
        df = pd.read_csv(
            str(path_to_annot / frame_name.replace(".jpg", "_rotated.csv")))

        # Drop all rows that have any NaN values
        df.dropna(inplace=True)

        # Drop all rows that have "Motorcycle" as "Type"
        df = df[df["Type"] != "Motorcycle"]

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

        crop_sub_imgs = []
        for bbox in crop_bboxes:
            crop_sub_imgs.append(
                image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]])

        crop_sub_annots = []
        for bbox in crop_bboxes:
            new_df = df[(df["x1"] >= bbox[0][0]) &
                        (df["x1"] <= bbox[1][0]) &
                        (df["y1"] >= bbox[0][1]) &
                        (df["y1"] <= bbox[1][1])].copy()
            # Change the coordinates of the bounding boxes to be relative to the sub image
            new_df['x1'] = new_df['x1'] - bbox[0][0]
            new_df['y1'] = new_df['y1'] - bbox[0][1]
            new_df['x2'] = new_df['x2'] - bbox[0][0]
            new_df['y2'] = new_df['y2'] - bbox[0][1]
            new_df['x3'] = new_df['x3'] - bbox[0][0]
            new_df['y3'] = new_df['y3'] - bbox[0][1]
            new_df['x4'] = new_df['x4'] - bbox[0][0]
            new_df['y4'] = new_df['y4'] - bbox[0][1]
            crop_sub_annots.append(new_df)

        # Draw the bounding boxes on corresponding sub images
        # for idx, sub_img in enumerate(crop_sub_imgs):
        #     for _, row in crop_sub_annots[idx].iterrows():
        #         cv2.line(sub_img, (row['x1'], row['y1']),
        #                  (row['x2'], row['y2']), (0, 255, 0), 2)
        #         cv2.line(sub_img, (row['x2'], row['y2']),
        #                  (row['x3'], row['y3']), (0, 255, 0), 2)
        #         cv2.line(sub_img, (row['x3'], row['y3']),
        #                  (row['x4'], row['y4']), (0, 255, 0), 2)
        #         cv2.line(sub_img, (row['x4'], row['y4']),
        #                  (row['x1'], row['y1']), (0, 255, 0), 2)

        # Save the sub images and corresponding annotations. For filename use the original filename with the sub image index appended
        for idx, sub_img in enumerate(crop_sub_imgs):
            out_file_name = frame_name.replace(".jpg",
                                               f"_{dataset_config['dateset_id']}_{idx}.jpg")
            cv2.imwrite(
                str(Path(dataset_config["output_path"]) / out_file_name),
                sub_img)

            out_lines.append(f"# {out_file_name}")
            for _, row in crop_sub_annots[idx].iterrows():
                upper_left = (min(row['x1'], row['x2'], row['x3'], row['x4']),
                              min(row['y1'], row['y2'], row['y3'], row['y4']))
                # upper_right = (max(row['x1'], row['x2'], row['x3'], row['x4']),
                #                min(row['y1'], row['y2'], row['y3'], row['y4']))
                # lower_left = (min(row['x1'], row['x2'], row['x3'], row['x4']),
                #               max(row['y1'], row['y2'], row['y3'], row['y4']))
                lower_right = (max(row['x1'], row['x2'], row['x3'], row['x4']),
                               max(row['y1'], row['y2'], row['y3'], row['y4']))
                middle_point = ((upper_left[0] + lower_right[0]) / 2,
                                (upper_left[1] + lower_right[1]) / 2)
                row_width = lower_right[0] - upper_left[0]
                row_height = lower_right[1] - upper_left[1]
                out_lines.append(f"{upper_left[0]} {upper_left[1]} "
                                 f"{row_width} {row_height} "
                                 f"{row['x1']} {row['y1']} 0.0 "
                                 f"{row['x2']} {row['y2']} 0.0 "
                                 f"{row['x3']} {row['y3']} 0.0 "
                                 f"{row['x4']} {row['y4']} 0.0 "
                                 f"{middle_point[0]} {middle_point[1]} ")

    final_line = "\n".join(out_lines)
    label_file = Path(dataset_config["output_path"]).parent / "label.txt"
    with open(label_file, "a") as f:
        if label_file.exists():
            f.write("\n")
        f.write(final_line)
