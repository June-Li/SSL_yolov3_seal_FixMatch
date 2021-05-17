import cv2
import numpy as np
import os


base_path = '/workspace/JuneLi/bbtv/SSL_yolov3_table_cell_FixMatch/data/ocr_table/semi/un_labels/train/'
label_name_list = os.listdir(base_path)
count_1 = 0
count_2 = 0
count_3 = 0
show_flag = 1
for label_name in label_name_list:
    lines = open(base_path + label_name, 'r').readlines()
    if len(lines) > 0:
        count_1 += 1
        if len(lines) > 1:
            count_2 += 1
        if len(lines) > 2:
            count_3 += 1
        if show_flag:
            show_img = cv2.imread(base_path.replace('/un_labels/', '/un_images/') + label_name.replace('.txt', '.jpg'))
            for line in lines:
                line = line.rstrip('\n').rstrip(' ').lstrip(' ').split(' ')
                [x_center, y_center, weight, height] = [float(i) for i in line[1:]]
                x_0 = int((x_center - weight / 2) * np.shape(show_img)[1])
                y_0 = int((y_center - height / 2) * np.shape(show_img)[0])
                x_1 = int((x_center + weight / 2) * np.shape(show_img)[1])
                y_1 = int((y_center + height / 2) * np.shape(show_img)[0])
                cv2.rectangle(show_img, (x_0, y_0), (x_1, y_1), (0, 0, 255), thickness=2)
            cv2.imwrite('buffer/a/' + label_name.replace('.txt', '.jpg'), show_img)
print('count_1 total num: ', count_1)
print('count_2 total num: ', count_2)
print('count_3 total num: ', count_3)
