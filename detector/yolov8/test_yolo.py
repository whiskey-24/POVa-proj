from ultralytics import YOLO
from pprint import pprint
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# vehicle_dict = {  # type : class_id
#     'Motorcycle': 0,  # Motorcycle
#     'Car': 1,  # Car
#     'Taxi': 1,  # Taxi
#     'Bus': 2,  # Bus
#     'Medium Vehicle': 3,  # Medium Vehicle
#     'Heavy Vehicle': 4,  # Heavy Vehicle
# }
vehicle_dict = {  # class_id : type
    0: 'Motorcycle',  # Motorcycle
    1: 'Car',  # Car
    2: 'Bus',  # Bus
    3: 'Medium Vehicle',  # Medium Vehicle
    4: 'Heavy Vehicle',  # Heavy Vehicle
}

vehicle_to_color = {
    'Motorcycle': (0, 0, 255),  # Motorcycle
    'Car': (0, 255, 0),  # Car
    'Taxi': (0, 255, 0),  # Taxi
    'Bus': (255, 0, 0),  # Bus
    'Medium Vehicle': (255, 255, 0),  # Medium Vehicle
    'Heavy Vehicle': (255, 0, 255),  # Heavy Vehicle
}

model = YOLO('runs/detect/train8/weights/best.pt')

cv2.namedWindow("test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("test", 640, 480)

if False:
    video_path = "/home/whiskey/Documents/2Mit/POVa/POVa-proj/map_reg/point_reg/2023_08_10_14_40_06.mp4"

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    paused = False
    first_frame = True
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            if first_frame:
                first_frame = False
                paused = True
            results = model(frame)  # Show the results
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                cv2.imshow("test", im_array)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord(" "):
            paused = not paused
else:
    img_path = "/home/whiskey/Documents/2Mit/POVa/POVa-proj/detector/data/20181029_D10_0900_0930/Frames/00001.jpg"
    frame = cv2.imread(img_path)
    results = model(frame)  # Show the results
    for r in results:
        boxes = r.boxes.cpu()
        boxes_coords = boxes.xyxy.numpy()
        boxes_scores = boxes.conf.numpy()
        boxes_cls = boxes.cls.numpy()
        for idx, box in enumerate(boxes_coords):
            box = box.astype(int)
            box_color = vehicle_to_color[vehicle_dict[int(boxes_cls[idx])]]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), box_color, 2)
        cv2.imshow("test", frame)

    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            break



