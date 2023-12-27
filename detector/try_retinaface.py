import cv2
import sys
sys.path.append(f"retinaface/Pytorch_Retinaface")
from retinaface.Pytorch_Retinaface.detect import VehicleDetector


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")


video_path = "/home/whiskey/Documents/2Mit/POVa/POVa-proj/map_reg/point_reg/2023_08_10_14_40_06.mp4"

# vehicle_detector = VehicleDetector("mobile0.25", "retinaface/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth",
#                                    False, 0.5, True)
vehicle_detector = VehicleDetector("resnet50", "retinaface/Pytorch_Retinaface/weights/Resnet50_Final.pth",
                                   False, 0.5, True)

cv2.namedWindow("test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("test", 640, 480)
cv2.setMouseCallback("test", mouse_callback)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

bbox = [(438, 60), (782, 338)]

# paused = False
# first_frame = True
# frame_num = 0
# while True:
#     if not paused:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_num += 1
#         # Crop frame to bbox
#         frame = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
#         if first_frame:
#             first_frame = False
#             paused = True
#         vehicles, draw_img = vehicle_detector.detect_image(frame)
#         cv2.putText(draw_img, f"Frame: {frame_num}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("test", draw_img)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
#     elif key == ord('p'):
#         paused = not paused
#     elif key == ord(" "):
#         paused = not paused

img_path = "/home/whiskey/Documents/2Mit/POVa/POVa-proj/detector/data/retinaface/images/00002_1_0.jpg"
img = cv2.imread(img_path)
vehicles, draw_img = vehicle_detector.detect_image(img)
cv2.imshow("test", draw_img)
while True:
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
