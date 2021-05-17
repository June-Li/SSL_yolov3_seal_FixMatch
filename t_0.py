import os
import argparse
import numpy as np
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class get_param():
    def __init__(self, model_path):
        self.source = './yolov3/data/images'
        # self.weights = model_path + '/table.pth'
        self.weights = '/workspace/JuneLi/bbtv/yolov3_train/save_model/1/best.pt'
        self.img_size = 640
        self.save_img = './yolov3/runs/detect/exp/'
        self.device = '0'
        self.augment = False
        self.conf_thres = 0.1
        self.iou_thres = 0.5


class Detector:
    def __init__(self, args):
        with torch.no_grad():
            self.opt = args
            self.source, self.weights, self.imgsz, self.save_img = self.opt.source, self.opt.weights, self.opt.img_size, self.opt.save_img
            self.device = select_device(self.opt.device)
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

            self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
            self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
            if self.half:
                self.model.half()  # to FP16
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def inference(self, path, image_name_list):
        with torch.no_grad():
            boxes_total = []
            confes_total = []
            clses_total = []
            t0 = time.time()

            imgs = []
            imgs_numpy = []
            for image_name in image_name_list:
                ii = cv2.imread(path + image_name)
                imgs_numpy.append(ii)
                # img = cv2.resize(ii, (self.imgsz, self.imgsz))
                img = letterbox(ii, new_shape=self.imgsz)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                imgs.append(img)

            img = np.array(imgs)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=self.opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                boxes = []
                confes = []
                clses = []
                p, s, im0 = Path(path + image_name_list[i]), '', imgs_numpy[i].copy()

                s += '%gx%g ' % imgs_numpy[i].shape[1:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(imgs_numpy[i].shape[1:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                        confes.append(float(conf.cpu().numpy()))
                        clses.append(int(cls.cpu().numpy()))
                        if self.opt.save_img != '':  # Add bbox to image
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
                    # cv2.imwrite('/workspace/JuneLi/bbtv/PaddleOCR-1.0-2021/result/inference_results-10/2_liushui/' + 'tabel_' + path.replace('.pdf', '.jpg'), np.array(im0, dtype=np.uint8))
                    # cv2.imwrite('./buffer/io.jpg', np.array(im0, dtype=np.uint8))
                    # print()
                    # time.sleep(99999)

                # Print time (inference + NMS)
                # print('%sdet table use time. (%.3fs)' % (s, t2 - t1))

                # Save results (image with detections)
                # if self.opt.save_img != '':
                #     if not os.path.exists(self.opt.save_img):
                #         os.mkdir(self.opt.save_img)
                #     cv2.imwrite(self.opt.save_img + p.name, im0)
                boxes_total.append(boxes)
                confes_total.append(confes)
                clses_total.append(clses)

        # print('det table use time. (%.3fs)' % (time.time() - t0))
        return boxes_total, confes_total, clses_total


def get_unlabel_label():
    detector = Detector(get_param(''))
    path = './data/ocr_table/images/train/'
    image_name_list = os.listdir(path)
    process_name_list = []
    for index_one, image_name in enumerate(image_name_list):
        process_name_list.append(image_name)
        if len(process_name_list) == 1 or index_one == len(image_name_list)-1:
            boxes_total, confes_total, clses_total = detector.inference(path, process_name_list)
            for index, process_name in enumerate(process_name_list):
                img = cv2.imread(path + process_name)
                for box in boxes_total[index]:
                    c1, c2 = (box[0], box[1]), (box[2], box[3])
                    cv2.rectangle(img, c1, c2, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
                cv2.imwrite('./buffer/output.jpg', img)
            process_name_list = []


if __name__ == '__main__':
    get_unlabel_label()
