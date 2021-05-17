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
    def __init__(self, model_path, conf_thres):
        self.weights = model_path + '/best.pt'
        self.img_size = 640
        self.device = '0'
        self.augment = False
        self.conf_thres = conf_thres
        self.iou_thres = 0.2


class Detector:
    def __init__(self, args, options):
        with torch.no_grad():
            self.opt = args
            self.weights, self.imgsz,  = self.opt.weights, self.opt.img_size
            self.device = select_device(self.opt.device)
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

            if os.path.exists(self.weights):
                self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
                print('load model ' + self.weights + '......')
            else:
                self.model = attempt_load(options.weights, map_location=self.device)  # load FP32 model
                print('load model ' + options.weights + '......')

            self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
            if self.half:
                self.model.half()  # to FP16
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def inference(self, path, im0s):
        with torch.no_grad():
            boxes = []
            confes = []
            clses = []
            t0 = time.time()

            img = letterbox(im0s, new_shape=self.imgsz)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = self.model(img, augment=self.opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = Path(path), '', im0s.copy()

                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                        confes.append(float(conf.cpu().numpy()))
                        clses.append(int(cls.cpu().numpy()))

        return boxes, confes, clses


def get_effective_radio(image, label):
    img_h, img_w, _ = np.shape(image)
    effective_area_list = []
    for line in label:
        line = line.rstrip('\n').rstrip(' ').lstrip(' ').split(' ')
        [x_center, y_center, weight, height] = [float(i) for i in line[1:]]
        x_0 = int((x_center - weight / 2) * img_w)
        y_0 = int((y_center - height / 2) * img_h)
        x_1 = int((x_center + weight / 2) * img_w)
        y_1 = int((y_center + height / 2) * img_h)
        effective_area_list.append(abs(x_1 - x_0) * abs(y_0 - y_1))
    effective_area = sum(effective_area_list)
    total_area = img_h * img_w
    radio = effective_area / total_area
    return radio


def gen_label_unlabel(model_path, options):
    detector = Detector(get_param(model_path, options.update_unlabel_conf), options)
    path = './data/ocr_table/semi/un_images/train/'
    image_name_list = os.listdir(path)
    for index_one, image_name in enumerate(image_name_list):
        image_path = path + image_name
        im0s = cv2.imread(image_path)
        boxes, confes, clses = detector.inference(image_path, im0s.copy())
        out_str = ''
        out_str_list = []
        for index, box in enumerate(boxes):
            img_h, img_w, _ = np.shape(im0s)
            x_center, y_center = (box[0] + box[2])/2/img_w, (box[1] + box[3])/2/img_h
            w, h = abs(box[0] - box[2])/img_w, abs(box[1] - box[3])/img_h
            out_str += '0 ' + ' '.join([str(round(i, 5)) for i in [x_center, y_center, w, h]]) + '\n'
            out_str_list.append('0 ' + ' '.join([str(round(i, 5)) for i in [x_center, y_center, w, h]]) + '\n')
            # c1, c2 = (box[0], box[1]), (box[2], box[3])
            # cv2.rectangle(im0s, c1, c2, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
            # cv2.putText(im0s, str(confes[index]), c1, cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
        if os.path.exists(image_path.replace('/un_images/', '/un_labels/').replace('.jpg', '.txt')):
            os.remove(image_path.replace('/un_images/', '/un_labels/').replace('.jpg', '.txt'))

        # radio = get_effective_radio(im0s.copy(), out_str_list)
        # if radio > 0.99:
        #     open(image_path.replace('/un_images/', '/un_labels/').replace('.jpg', '.txt'), 'w').writelines(out_str)
        # else:
        #     open(image_path.replace('/un_images/', '/un_labels/').replace('.jpg', '.txt'), 'w').writelines('')
        if options.train_type == 'SSL':
            open(image_path.replace('/un_images/', '/un_labels/').replace('.jpg', '.txt'), 'w').writelines(out_str)
        elif options.train_type == 'SL':
            open(image_path.replace('/un_images/', '/un_labels/').replace('.jpg', '.txt'), 'w').writelines('')
        else:
            print('请重新输入train_type参数的值')
            raise ValueError
        # cv2.imwrite('./buffer/a/' + image_name, im0s)
        # print(boxes)
        if index_one % 500 == 0:
            print('processed num: ', index_one)

