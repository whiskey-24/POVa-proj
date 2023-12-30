from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
from dataclasses import dataclass
from custom_dataclasess import Vehicle
import math


# parser = argparse.ArgumentParser(description='Retinaface')
#
# # -m ./weights/mobilenet0.25_epoch_160.pth
#
# parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
#                     type=str, help='Trained state_dict file path to open')
# parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
# parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
# parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
# parser.add_argument('--top_k', default=5000, type=int, help='top_k')
# parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
# parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
# parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
# parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
# args = parser.parse_args()
def calculate_angle(center, orientation):
    # Calculate the vector from center to orientation
    vector_x = orientation[0] - center[0]
    vector_y = center[1] - orientation[1]  # Flip the y-coordinates to make 0 degrees represent upward

    # Calculate the angle using arctangent
    angle_rad = math.atan2(vector_x, vector_y)

    # Convert radians to degrees
    angle_deg = math.degrees(angle_rad + math.pi * 2)

    # Adjust the angle to be in the range [0, 360)
    # angle_deg = (angle_deg + 360) % 360
    angle_deg = angle_deg % 360

    return angle_deg


class VehicleDetector:

    def __init__(self, network: str, model: str, cpu: bool, vis_thres: float, draw: bool,
                 confidence_threshold: float = 0.5, top_k: int = 5000,
                 nms_threshold: float = 0.4, keep_top_k: int = 750):
        self.network = network
        self.model = model
        self.cpu = cpu
        self.vis_thres = vis_thres
        self.draw = draw
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k

        torch.set_grad_enabled(False)
        self.cfg = None
        if network == "mobile0.25":
            self.cfg = cfg_mnet
        elif network == "resnet50":
            self.cfg = cfg_re50
        # net and model
        net = RetinaFace(cfg=self.cfg, phase='test')
        net = load_model(net, model, cpu)
        net.eval()
        print('Finished loading model!')
        # print(net)
        cudnn.benchmark = True
        self.device = torch.device("cpu" if cpu else "cuda")
        self.net = net.to(self.device)

        self.resize = 1

    def _detect(self, img_in: np.ndarray) -> list[Vehicle] | tuple[list[Vehicle], np.ndarray]:
        img_raw = img_in.copy()
        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        # tic = time.time()
        loc, conf, landms = self.net(img)  # forward pass
        # print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data,
                              self.cfg['variance'])
        scale1 = torch.Tensor(
            [img.shape[3], img.shape[2], img.shape[3], img.shape[2],
             img.shape[3], img.shape[2], img.shape[3], img.shape[2],
             img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32,
                                                                copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        faces: list[Vehicle] = []
        if self.draw:
            for idx, b in enumerate(dets):
                if b[4] < self.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255),
                              2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (255, 0, 0), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 255, 255), 4)
                landmarks = [(b[5], b[6]), (b[7], b[8]), (b[9], b[10]),
                             (b[11], b[12]), (b[13], b[14])]
                centre_x = (b[0] + b[2]) / 2
                centre_y = (b[1] + b[3]) / 2
                orientation_x = (b[9] + b[11]) / 2
                orientation_y = (b[10] + b[12]) / 2
                angle = calculate_angle((centre_x, centre_y),
                                        (orientation_x, orientation_y))

                # Draw a pink line from centre in direction of orientation (based on angle) that has length max(width, height)
                length = max(b[2] - b[0], b[3] - b[1])
                P2x = int(round(centre_x + length * math.cos(math.radians(angle) - math.pi / 2)))
                P2y = int(round(centre_y + length * math.sin(math.radians(angle) - math.pi / 2)))
                cv2.line(img_raw, (int(centre_x), int(centre_y)), (P2x, P2y), (255, 0, 255), 2)

                faces.append(Vehicle(b[0], b[1], b[2], b[3], b[4], landmarks, angle))
            # save image

            return faces, img_raw
        else:
            for idx, b in enumerate(dets):
                if b[4] < self.vis_thres:
                    continue
                landmarks = [(b[5], b[6]), (b[7], b[8]), (b[9], b[10]),
                             (b[11], b[12]), (b[13], b[14])]
                centre_x = (b[0] + b[2]) / 2
                centre_y = (b[1] + b[3]) / 2
                orientation_x = (b[9] + b[11]) / 2
                orientation_y = (b[10] + b[12]) / 2
                angle = calculate_angle((centre_x, centre_y),
                                        (orientation_x, orientation_y))
                faces.append(Vehicle(b[0], b[1], b[2], b[3], b[4], landmarks, angle))
            return faces

    def detect_path(self, path_to_img: str) -> list[Vehicle] | tuple[list[Vehicle], np.ndarray]:
        img_raw = cv2.imread(path_to_img, cv2.IMREAD_COLOR)
        return self._detect(img_raw)

    def detect_image(self, img: np.ndarray) -> list[Vehicle] | tuple[list[Vehicle], np.ndarray]:
        return self._detect(img)


def detect_face(network: str, model: str, cpu: bool, image_path: str, vis_thres: float, draw: bool = True, image: np.ndarray = None,
                confidence_threshold: float = 0.5, top_k: int = 5000,
                nms_threshold: float = 0.4, keep_top_k: int = 750) -> list[Vehicle] | tuple[list[Vehicle], np.ndarray]:
    torch.set_grad_enabled(False)
    cfg = None
    if network == "mobile0.25":
        cfg = cfg_mnet
    elif network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, model, cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if cpu else "cuda")
    net = net.to(device)

    resize = 1

    if image is not None:
        img_raw = image
    else:
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor(
        [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    print('net forward time: {:.4f}'.format(time.time() - tic))

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor(
        [img.shape[3], img.shape[2], img.shape[3], img.shape[2],
         img.shape[3], img.shape[2], img.shape[3], img.shape[2],
         img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32,
                                                            copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    faces: list[Vehicle] = []
    if draw:
        for idx, b in enumerate(dets):
            if b[4] < vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            landmarks = [(b[5], b[6]), (b[7], b[8]), (b[9], b[10]), (b[11], b[12]), (b[13], b[14])]
            faces.append(Vehicle(b[0], b[1], b[2], b[3], b[4], landmarks))
        # save image

        return faces, img_raw
    else:
        for idx, b in enumerate(dets):
            if b[4] < vis_thres:
                continue
            landmarks = [(b[5], b[6]), (b[7], b[8]), (b[9], b[10]), (b[11], b[12]), (b[13], b[14])]
            faces.append(Vehicle(b[0], b[1], b[2], b[3], b[4], landmarks))
        return faces




def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

#
# if __name__ == '__main__':
#     torch.set_grad_enabled(False)
#     cfg = None
#     if args.network == "mobile0.25":
#         cfg = cfg_mnet
#     elif args.network == "resnet50":
#         cfg = cfg_re50
#     # net and model
#     net = RetinaFace(cfg=cfg, phase = 'test')
#     net = load_model(net, args.trained_model, args.cpu)
#     net.eval()
#     print('Finished loading model!')
#     # print(net)
#     cudnn.benchmark = True
#     device = torch.device("cpu" if args.cpu else "cuda")
#     net = net.to(device)
#
#     resize = 1
#
#     # testing begin
#     for i in range(100):
#         # image_path = "./curve/test.jpg"
#         image_path = "/home/whiskey/Documents/2Mit/POVa/POVa-proj/detector/data/retinaface/images/00001_1_0.jpg"
#         img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
#
#         img = np.float32(img_raw)
#
#         im_height, im_width, _ = img.shape
#         scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
#         img -= (104, 117, 123)
#         img = img.transpose(2, 0, 1)
#         img = torch.from_numpy(img).unsqueeze(0)
#         img = img.to(device)
#         scale = scale.to(device)
#
#         tic = time.time()
#         loc, conf, landms = net(img)  # forward pass
#         print('net forward time: {:.4f}'.format(time.time() - tic))
#
#         priorbox = PriorBox(cfg, image_size=(im_height, im_width))
#         priors = priorbox.forward()
#         priors = priors.to(device)
#         prior_data = priors.data
#         boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
#         boxes = boxes * scale / resize
#         boxes = boxes.cpu().numpy()
#         scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
#         landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
#         scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                                img.shape[3], img.shape[2]])
#         scale1 = scale1.to(device)
#         landms = landms * scale1 / resize
#         landms = landms.cpu().numpy()
#
#         # ignore low scores
#         inds = np.where(scores > args.confidence_threshold)[0]
#         boxes = boxes[inds]
#         landms = landms[inds]
#         scores = scores[inds]
#
#         # keep top-K before NMS
#         order = scores.argsort()[::-1][:args.top_k]
#         boxes = boxes[order]
#         landms = landms[order]
#         scores = scores[order]
#
#         # do NMS
#         dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
#         keep = py_cpu_nms(dets, args.nms_threshold)
#         # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
#         dets = dets[keep, :]
#         landms = landms[keep]
#
#         # keep top-K faster NMS
#         dets = dets[:args.keep_top_k, :]
#         landms = landms[:args.keep_top_k, :]
#
#         dets = np.concatenate((dets, landms), axis=1)
#
#         # show image
#         if args.save_image:
#             for b in dets:
#                 if b[4] < args.vis_thres:
#                     continue
#                 text = "{:.4f}".format(b[4])
#                 b = list(map(int, b))
#                 cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
#                 cx = b[0]
#                 cy = b[1] + 12
#                 cv2.putText(img_raw, text, (cx, cy),
#                             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
#
#                 # landms
#                 cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
#                 cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
#                 cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
#                 cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
#                 cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
#             # save image
#
#             name = "test.jpg"
#             # cv2.imwrite(name, img_raw)
#             cv2.imshow("test", img_raw)
#             while True:
#                 key = cv2.waitKey(1)
#                 if key == ord('q'):
#                     break
