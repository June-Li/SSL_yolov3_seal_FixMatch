import os
import shutil

# base_path = '/workspace/JuneLi/bbtv/SSL_yolov3_FixMatch/data/ocr_table/small/'
# in_image_path = base_path + 'images/train/'
# in_label_path = base_path + 'labels/train/'
#
# image_name_list = os.listdir(in_image_path)
# for index, image_name in enumerate(image_name_list):
#     if index % 2 != 1:
#         os.remove(in_image_path + image_name)
#         os.remove(in_label_path + image_name.replace('.jpg', '.txt'))
# print(len(image_name_list))

base_path = '/workspace/JuneLi/bbtv/SSL_yolov3_FixMatch/data/ocr_table/small/'
in_image_path = base_path + 'images/train/'
in_label_path = base_path + 'labels/train/'

image_name_list = os.listdir(in_image_path)
for i in range(1, 100):
    for index, image_name in enumerate(image_name_list):
        shutil.copy(in_image_path + image_name, in_image_path + str(i) + '_' + image_name)
        shutil.copy(in_label_path + image_name.replace('.jpg', '.txt'), in_label_path + str(i) + '_' + image_name.replace('.jpg', '.txt'))
