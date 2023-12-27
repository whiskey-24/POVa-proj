import cv2
import numpy as np
import matplotlib.cm as cm
import torch
from pathlib import Path

from map_reg.point_reg.SuperGlueModel.models.matching import Matching
from map_reg.point_reg.SuperGlueModel.models.utils import read_image, make_matching_plot_fast

torch.set_grad_enabled(False)


class ImageMatcher:

    def __init__(self, sg_resize=None, sg_resize_float=True,
                 sg_nms_radius=4, sg_keypoint_threshold=0.005,
                 sg_max_keypoints=1024, sg_superglue='outdoor',
                 sg_sinkhorn_iterations=20, sg_match_threshold=0.2,
                 sg_force_cpu=False, rc_reproj_threshold=0.001,
                 rc_confidence=0.999999, rc_max_iters=10000,
                 draw=False, tmp_path="tmp"):
        if sg_resize is None:
            sg_resize = [640, 480]

        # SuperGlue parameters
        self.sg_resize = sg_resize
        self.sg_resize_float = sg_resize_float
        self.sg_nms_radius = sg_nms_radius
        self.sg_keypoint_threshold = sg_keypoint_threshold
        self.sg_max_keypoints = sg_max_keypoints
        self.sg_superglue = sg_superglue
        self.sg_sinkhorn_iterations = sg_sinkhorn_iterations
        self.sg_match_threshold = sg_match_threshold
        self.sg_force_cpu = sg_force_cpu
        self.mkpts0: np.ndarray | None = None
        self.mkpts1: np.ndarray | None = None

        # RANSAC parameters
        self.rc_reproj_threshold = rc_reproj_threshold
        self.rc_confidence = rc_confidence
        self.rc_max_iters = rc_max_iters

        # Paths
        self.sat_path = ""
        self.drone_path = ""
        self.tmp_path = tmp_path
        Path(tmp_path).mkdir(parents=True, exist_ok=True)

        # Drawing parameters and images
        self.draw = draw
        self.sat_img: np.ndarray | None = None
        self.drone_img: np.ndarray | None = None
        self.transformed_sat: np.ndarray | None = None
        self.super_glue_img: np.ndarray | None = None
        self.projection_img: np.ndarray | None = None

        # Transformation matrices
        self.affine_matrix: np.ndarray | None = None
        self.project_matrix: np.ndarray | None = None

    def rotate_and_scale(self, img1: np.ndarray, points_im1: np.ndarray,
                         points_im2: np.ndarray,
                         output_size: tuple[int, int]) -> None:
        # Convert images to grayscale
        # if len(img1.shape) > 2:
        #     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # else:
        #     gray1 = img1

        # Find transformation matrix
        self.affine_matrix = cv2.getAffineTransform(points_im1, points_im2)

        # Apply the transformation to img1
        self.transformed_sat = cv2.warpAffine(img1, self.affine_matrix,
                                              (output_size[0], output_size[1]))

    def transform_sat_img(self, sat_img: np.ndarray, drone_img: np.ndarray,
                          corner_points: list[list[int]],
                          # [[top_left, bottom_right, bottom_left]]
                          add_percent: float) -> None:
        drone_shape = drone_img.shape
        drone_height, drone_width = drone_shape[0], drone_shape[1]

        point1 = np.array(corner_points, dtype=np.float32)
        point2 = np.array(
            [[drone_height * add_percent, drone_height * add_percent],
             [drone_width * (1 - add_percent),
              drone_height * (1 - add_percent)],
             [drone_width * add_percent,
              drone_height * (1 - add_percent)]], dtype=np.float32)

        output_size = (int(drone_width * (1 + 2 * add_percent)),
                       int(drone_height * (1 + 2 * add_percent)))

        # Apply transformation
        self.rotate_and_scale(sat_img, point1, point2, output_size)

    # Base code is from SuperGlueModel/match_points.py
    # https://github.com/magicleap/SuperGluePretrainedNetwork
    def superpoints_superglue_match(self, path_to_img1: str,
                                    path_to_img2: str):
        # force_cpu = False
        device = 'cuda' if torch.cuda.is_available() and not self.sg_force_cpu else 'cpu'

        rot0, rot1 = 0, 0
        config = {
            'superpoint': {
                'nms_radius': self.sg_nms_radius,
                'keypoint_threshold': self.sg_keypoint_threshold,
                'max_keypoints': self.sg_max_keypoints
            },
            'superglue': {
                'weights': self.sg_superglue,
                'sinkhorn_iterations': self.sg_sinkhorn_iterations,
                'match_threshold': self.sg_match_threshold,
            }
        }
        matching = Matching(config).eval().to(device)

        image0, inp0, scales0 = read_image(path_to_img1, device,
                                           self.sg_resize, rot0,
                                           self.sg_resize_float)
        image1, inp1, scales1 = read_image(path_to_img2, device,
                                           self.sg_resize, rot1,
                                           self.sg_resize_float)

        if image0 is None or image1 is None:
            print(f"Problem reading image pair: {path_to_img1} {path_to_img2}")
            exit(1)

        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                       'matches': matches, 'match_confidence': conf}

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        self.mkpts0 = mkpts0
        self.mkpts1 = mkpts1

        if self.draw:
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
            ]

            # make_matching_plot(
            #     image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
            #     text, viz_path, opt.show_keypoints,
            #     opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            out_img = make_matching_plot_fast(image0, image1, kpts0, kpts1,
                                              mkpts0,
                                              mkpts1,
                                              color, text, path=None,
                                              show_keypoints=True,
                                              margin=10,
                                              opencv_display=False,
                                              small_text=small_text)

            # cv2.imshow('Matched img', out_img)
            # while True:
            #     key = cv2.waitKey(0)
            #     if key == ord('q'):
            #         break
            self.super_glue_img = out_img

    def convert_transformed_coord_to_orig(self, x: int, y: int) -> tuple[
        int, int]:
        # Convert the x, y coordinates into a homogeneous coordinate
        transformed_coord = np.array([x, y, 1])
        # Compute the inverse of the transformation matrix
        inv_transformed_matrix = cv2.invertAffineTransform(self.affine_matrix)
        # Use the inverse transformation matrix to map the point back to the original image
        original_coord = np.dot(inv_transformed_matrix, transformed_coord)
        return int(original_coord[0]), int(original_coord[1])

    def load_imgs(self, sat_path: str | np.ndarray,
                  drone_path: str | np.ndarray,
                  corner_points: list[list[int]], add_percent: float):
        if isinstance(sat_path, str):
            self.sat_path = sat_path
            self.drone_path = drone_path

            self.sat_img = cv2.imread(sat_path)
            self.drone_img = cv2.imread(drone_path)
        else:
            self.sat_path = ""
            self.drone_path = ""

            self.sat_img = sat_path
            self.drone_img = drone_path

        self.transform_sat_img(self.sat_img, self.drone_img, corner_points,
                               add_percent)

    def draw_all_imgs(self, show=True):
        if self.sat_img is not None:
            cv2.imshow("Satellite image", self.sat_img)
        if self.drone_img is not None:
            cv2.imshow("Drone image", self.drone_img)
        if self.transformed_sat is not None:
            cv2.imshow("Transformed satellite image", self.transformed_sat)
        if self.super_glue_img is not None:
            cv2.imshow("SuperGlue matches", self.super_glue_img)
        if self.projection_img is not None:
            cv2.imshow("Projection", self.projection_img)

        if show:
            while True:
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break

    # Parts of these functions are from OpenCV docs
    # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    def process_img(self, drone_img: np.ndarray):
        transformed_path = f"{self.tmp_path}_transformed.png"
        cv2.imwrite(transformed_path, cv2.cvtColor(self.transformed_sat,
                                                   cv2.COLOR_BGR2GRAY))

        tmp_drone_path = f"{self.tmp_path}_drone.png)"
        cv2.imwrite(tmp_drone_path,
                    cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY))

        self.superpoints_superglue_match(tmp_drone_path, transformed_path)

        self.project_matrix, mask = cv2.findHomography(self.mkpts0,
                                                       self.mkpts1, cv2.RANSAC,
                                                       ransacReprojThreshold=self.rc_reproj_threshold,
                                                       confidence=self.rc_confidence,
                                                       maxIters=self.rc_max_iters)

        # matchesMask = mask.ravel().tolist()
        drone_shape = drone_img.shape
        h, w = drone_shape[0], drone_shape[1]
        pts = np.float32(
            [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, self.project_matrix)

        # TODO check if dst is near border and if so, move transformed_sat

        if self.draw:
            self.projection_img = cv2.polylines(img=self.transformed_sat.copy(),
                                                pts=[np.int32(dst)], isClosed=True,
                                                color=[255, 255, 255], thickness=3,
                                                lineType=cv2.LINE_AA)
            self.draw_all_imgs(show=True)


# def mouse_callback_from_transf_to_orig(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         original_coord = convert_transformed_coord_to_orig(x, y)
#
#         # Draw a circle on the original image
#         draw = sat_img_draw.copy()
#         cv2.circle(draw, (original_coord[0], original_coord[1]),
#                    5, (0, 0, 255), -1)
#         cv2.imshow('Draw', draw)
#
#         print(f"Original coordinates: {original_coord}")
#
#
# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Convert the x, y coordinates into a homogeneous coordinate
#         transformed_coord = np.array([x, y, 1])
#
#         # Compute the inverse of the transformation matrix
#         inv_transformed_matrix = np.linalg.pinv(M)
#
#         # Use the inverse transformation matrix to map the point back to the original image
#         original_coord = np.dot(M, transformed_coord)
#
#         # Convert back to non-homogeneous coordinates
#         original_coord = original_coord[:2] / original_coord[2]
#         draw = img_2_draw.copy()
#         cv2.circle(draw, (int(original_coord[0]), int(original_coord[1])),
#                    5, (0, 0, 255), -1)
#         cv2.imshow('Draw2', draw)
#
#         print(f"Original coordinates: {original_coord}")


if __name__ == '__main__':
    top_left = [599, 325]
    bottom_right = [205, 251]
    bottom_left = [491, 129]
    matcher = ImageMatcher(draw=True, rc_reproj_threshold=2)
    matcher.load_imgs("imgs/sat_img.png", "imgs/from_drone.jpg",
                      [top_left, bottom_right, bottom_left], 0.1)
    matcher.process_img(drone_img=matcher.drone_img)
