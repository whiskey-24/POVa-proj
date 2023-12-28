import cv2
import sys
import numpy as np
from custom_dataclasess import Vehicle

# Detector
sys.path.append(f"detector/retinaface/Pytorch_Retinaface")
from detector.retinaface.Pytorch_Retinaface.detect import VehicleDetector

# Map registration
from map_reg.satellite_extractor.satellite_extractor import SatelliteExtractor, \
    TiledImage
from map_reg.point_reg.super_glue import ImageMatcher


class Application:

    def __init__(self, det_backend_type: str, det_model_path: str,
                 det_model_threshold: float, det_use_cpu: bool,
                 match_corner_points: list[list[int]],
                 map_lat: float, map_lon: float, draw: bool = True, debug: bool = False):
        self.draw = draw
        # Prepare windows opened by application
        if self.draw:
            cv2.namedWindow("Map", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Map", 640, 480)
        self.debug = debug

        self.vehicle_detector = VehicleDetector(det_backend_type,
                                                det_model_path,
                                                det_use_cpu,
                                                det_model_threshold, self.draw)

        self.sat_extractor = SatelliteExtractor()
        self.init_image = self.sat_extractor.download_img_range(lat=map_lat,
                                                                lon=map_lon,
                                                                x_range=3,
                                                                y_range=3,
                                                                stitch=True)

        self.image_matcher = ImageMatcher(draw=self.debug, show=False)
        self.init_corner_points = match_corner_points

    def process_frame(self, img: np.ndarray) -> None | np.ndarray:
        if self.draw:
            vehicles, draw_img = self.vehicle_detector.detect_image(img)
        else:
            vehicles = self.vehicle_detector.detect_image(img)

        self.image_matcher.load_imgs(self.init_image.get_out_img(), img,
                                     self.init_corner_points, 0.05)
        self.image_matcher.process_img(drone_img=self.image_matcher.drone_img)

        if self.draw:
            self.draw_vehicles_on_map(vehicles)
            return draw_img
        else:
            return None

    def draw_vehicles_on_map(self, vehicles: list[Vehicle]) -> None:
        sat_copy = self.init_image.get_out_img().copy()
        vehicle_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
                          (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        for idx, vehicle in enumerate(vehicles):
            middle_point = [int((vehicle.x1 + vehicle.x2) / 2),
                            int((vehicle.y1 + vehicle.y2) / 2)]
            map_point = self.image_matcher.drone_to_sat(middle_point[0],
                                                        middle_point[1])
            lat, lon = self.init_image.get_lat_lon_from_xy(map_point[0],
                                                           map_point[1])

            color_idx = idx % len(vehicle_colors)
            color = vehicle_colors[color_idx]
            cv2.circle(sat_copy, map_point, 7, color, -1)
            cv2.putText(sat_copy, f"Lat: {lat:.4f}", (map_point[0] + 10, map_point[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(sat_copy, f"Lon: {lon:.4f}", (map_point[0] + 10, map_point[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Map", sat_copy)


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")


if __name__ == "__main__":
    # centre = (300, 300)
    # up = (300, 0)
    # right = (600, 300)
    # down = (300, 600)
    # left = (0, 300)
    # all_dir = {"up": up, "right": right, "down": down, "left": left}
    # for dir, point in all_dir.items():
    #     angle = calculate_angle(centre, point)
    #     all_dir[dir] = round(angle)
    # print(f"-------{all_dir['up']}-------")
    # print(f"{all_dir['left']}-------{all_dir['right']}")
    # print(f"-------{all_dir['down']}-------")
    # for dir, angle in all_dir.items():
    #     P2x = int(round(centre[0] + 2 * -math.cos(math.radians(angle))))
    #     P2y = int(round(centre[1] + 2 * math.sin(math.radians(angle))))
    #     all_dir[dir] = (P2x, P2y)
    # print(f"-------{all_dir['up']}-------")
    # print(f"{all_dir['left']}-------{all_dir['right']}")
    # print(f"-------{all_dir['down']}-------")
    # exit(0)
    # cv2.namedWindow("Application", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Application", 640, 480)
    # cv2.setMouseCallback("Application", mouse_callback)
    #
    # while True:
    #     key = cv2.waitKey(1)
    #     if key == ord('q'):
    #         exit(0)
    resize = 1
    # top_left = [599 // resize, 325 // resize]
    # bottom_right = [205 // resize, 251 // resize]
    # bottom_left = [491 // resize, 129 // resize]
    top_left = [1238 // resize, 1287 // resize]
    bottom_right = [871 // resize, 496 // resize]
    bottom_left = [1524 // resize, 946 // resize]
    super_glue_init = [top_left, bottom_right, bottom_left]

    app = Application("resnet50",
                      "detector/retinaface/Pytorch_Retinaface/weights/Resnet50_Final.pth",
                      0.5, False, super_glue_init, 48.37017393185813,
                      17.494309514887174)

    cap = cv2.VideoCapture("map_reg/point_reg/2023_08_10_14_40_06.mp4")

    cv2.namedWindow("Application", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Application", 640, 480)
    cv2.setMouseCallback("Application", mouse_callback)

    paused = False
    frame_num = 0
    frame_hop = True
    skip_to_frame = 400
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            if frame_num < skip_to_frame:
                continue
            if frame_hop:
                paused = True
                frame_hop = False
            if resize != 1:
                frame = cv2.resize(frame, (0, 0), fx=1 / resize, fy=1 / resize)
            proc_img = app.process_frame(frame)
            cv2.putText(proc_img, f"Frame: {frame_num}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Application", proc_img)
            # cv2.imshow("Clean frame", frame)
            # cv2.imshow("Application", app.init_image.get_out_img())
            # cv2.imshow("Application frame", frame)

        # cv2.waitKey(1)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord(" "):
            paused = not paused
        elif key == ord("n"):
            frame_hop = True
            paused = False
